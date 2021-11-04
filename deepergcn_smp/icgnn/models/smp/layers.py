import torch
import torch.nn as nn
from torch import Tensor as Tensor
from torch.nn import Linear as Linear
import torch.nn.init as init
from torch.nn.init import _calculate_correct_fan, calculate_gain
import torch.nn.functional as F
from torch_geometric.nn import (
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    MessagePassing,
)
import math

small_gain = 0.01


def pooling(x: torch.Tensor, batch_info, method):
    if method == "add":
        return global_add_pool(x, batch_info["batch"], batch_info["num_graphs"])
    elif method == "mean":
        return global_mean_pool(x, batch_info["batch"], batch_info["num_graphs"])
    elif method == "max":
        return global_max_pool(x, batch_info["batch"], batch_info["num_graphs"])
    else:
        raise ValueError("Pooling method not implemented")


def kaiming_init_with_gain(
    x: Tensor, gain: float, a=0, mode="fan_in", nonlinearity="relu"
):
    fan = _calculate_correct_fan(x, mode)
    non_linearity_gain = calculate_gain(nonlinearity, a)
    std = non_linearity_gain / math.sqrt(fan)
    bound = (
        math.sqrt(3.0) * std * gain
    )  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return x.uniform_(-bound, bound)


class BatchNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)

    def reset_parameters(self):
        self.bn.reset_parameters()

    def forward(self, u):
        return self.bn(u.transpose(1, 2)).transpose(1, 2)


class EdgeCounter(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index, batch, batch_size):
        n_edges = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return global_mean_pool(n_edges, batch, batch_size)[batch]


class Linear(nn.Module):
    """Linear layer with potentially smaller parameters at initialization."""

    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(self, in_features, out_features, bias=True, gain: float = 1.0):
        super().__init__()
        self.gain = gain
        self.lin = nn.Linear(in_features, out_features, bias)

    def reset_parameters(self):
        kaiming_init_with_gain(self.lin.weight, self.gain)
        if self.lin.bias is not None:
            nn.init.normal_(
                self.lin.bias, 0, self.gain / math.sqrt(self.lin.out_features)
            )

    def forward(self, x):
        return self.lin.forward(x)


class EntrywiseU(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_towers=None):
        super().__init__()
        if num_towers is None:
            num_towers = in_features
        self.lin1 = torch.nn.Conv1d(
            in_features, out_features, kernel_size=1, groups=num_towers, bias=False
        )

    def forward(self, u):
        """u: N x colors x channels."""
        u = u.transpose(1, 2)
        u = self.lin1(u)
        return u.transpose(1, 2)



class UtoU(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, residual=True, n_groups=None
    ):
        super().__init__()
        if n_groups is None:
            n_groups = 1
        self.residual = residual
        self.lin1 = torch.nn.Conv1d(
            in_features, out_features, kernel_size=1, groups=n_groups, bias=True
        )
        self.lin2 = torch.nn.Conv1d(
            in_features, out_features, kernel_size=1, groups=n_groups, bias=False
        )
        self.lin3 = torch.nn.Conv1d(
            in_features, out_features, kernel_size=1, groups=n_groups, bias=False
        )

    def forward(self, u: Tensor, batch_info: dict = None):
        """U: N x n_colors x channels"""
        old_u = u
        n = batch_info["num_nodes"]
        num_colors = u.shape[1]
        out_feat = self.lin1.out_channels

        mask = batch_info["mask"][..., None].expand(n, num_colors, out_feat)
        normalizer = batch_info["n_batch"]
        mean2 = torch.sum(u / normalizer, dim=1)  # N, in_feat
        mean2 = mean2.unsqueeze(-1)  # N, in_feat, 1
        # 1. Transform u element-wise
        u = u.permute(0, 2, 1)  # In conv1d, channel dimension is second
        out = self.lin1(u).permute(0, 2, 1)

        # 2. Put in self of each line the sum over each line
        # The 0.1 factor is here to bias the network in favor of learning powers of the adjacency
        z2 = self.lin2(mean2) * 0.1  # N, out_feat, 1
        z2 = z2.transpose(1, 2)  # N, 1, out_feat
        index_tensor = batch_info["coloring"][:, :, None].expand(
            out.shape[0], 1, out_feat
        )
        out.scatter_add_(1, index_tensor, z2)  # n, n_colors, out_feat

        # 3. Put everywhere the sum over each line
        z3 = self.lin3(mean2)  # N, out_feat, 1
        z3 = z3.transpose(1, 2)  # N, 1, out_feat
        out3 = z3.expand(n, num_colors, out_feat)
        out += out3 * mask * 0.1  # Mask the extra colors
        if self.residual:
            return old_u + out
        return out


class UtoGlobal(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, gain: float):
        super().__init__()
        self.lin1 = Linear(in_features, out_features, bias, gain=gain)
        self.lin2 = Linear(in_features, out_features, bias, gain=gain)

    def reset_parameters(self):
        for layer in [self.lin1, self.lin2]:
            layer.reset_parameters()

    def forward(self, u, batch_info: dict, method="mean"):
        """u: (num_nodes, colors, in_features)
        output: (batch_size, out_features)."""
        coloring = batch_info["coloring"]
        # Extract trace
        index_tensor = coloring[:, :, None].expand(u.shape[0], 1, u.shape[2])
        extended_diag = u.gather(1, index_tensor)[:, 0, :]  # n_nodes, in_feat
        mean_batch_trace = pooling(
            extended_diag, batch_info, "mean"
        )  # n_graphs, in_feat
        out1 = self.lin1(mean_batch_trace)  # bs, out_feat
        # Extract sum of elements - trace
        mean = torch.sum(u / batch_info["n_batch"], dim=1)  # num_nodes, in_feat
        batch_sum = pooling(mean, batch_info, "mean")  # n_graphs, in_feat
        batch_sum = batch_sum - mean_batch_trace  # make the basis orthogonal
        out2 = self.lin2(batch_sum)  # bs, out_feat
        return out1 + out2



class GraphExtractor(nn.Module):
    def __init__(
        self, in_features: int, out_features: int
    ):
        super().__init__()
        self.extractor = UtoGlobal(
            in_features, out_features, True, 1
        )
        self.lin = nn.Linear(out_features, out_features)

    def reset_parameters(self):
        for layer in [self.extractor, self.lin]:
            layer.reset_parameters()

    def forward(self, u: Tensor, batch_info: dict):
        out = self.extractor(u, batch_info)
        out = out + self.lin(F.relu(out))
        return out
