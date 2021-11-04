import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from icgnn.models.smp.layers import UtoU, EntrywiseU


class SMPLayer(MessagePassing):
    def __init__(
        self,
        in_features: int,
        num_towers: int,
        out_features: int,
        edge_features: int,
    ):
        """Use a MLP both for the update and message function + edge features."""
        super().__init__(aggr="add", node_dim=-3)
        self.in_u, self.out_u, self.edge_features = (
            in_features,
            out_features,
            edge_features,
        )
        self.edge_nn = (
            nn.Linear(edge_features, out_features) 
        )

        self.message_nn = UtoU(
            in_features, out_features, n_groups=num_towers, residual=False
        )

        args_order2 = [out_features, out_features, num_towers]
        entry_wise = EntrywiseU
        self.order2_i = entry_wise(*args_order2)
        self.order2_j = entry_wise(*args_order2)
        self.order2 = entry_wise(*args_order2)

        self.update1 = nn.Linear(2 * out_features, out_features)
        self.update2 = nn.Linear(out_features, out_features)

    def forward(self, u, edge_index, edge_attr, batch_info, dist_basis=None):
        n = batch_info["num_nodes"]
        u = self.message_nn(u, batch_info)
        u1 = self.order2_i(u)
        u2 = self.order2_j(u)

        if dist_basis is not None:
            # add an extra dimension to dist basis
            dist_basis = dist_basis.unsqueeze(1)

        new_u = self.propagate(
            edge_index,
            size=(n, n),
            u=u,
            u1=u1,
            u2=u2,
            edge_attr=edge_attr,
            dist_basis=dist_basis,
        )
        new_u /= batch_info["average_edges"]
        return new_u

    def message(self, u_j, u1_i, u2_j, edge_attr, dist_basis_i=None):
        """
        dist_basis_i: extra feature of the node in the linegraph
        """
        if dist_basis_i is not None:
            # multiply the edge attr with the target node's dist basis
            # remove the extra dim from dist basis
            edge_attr = edge_attr * dist_basis_i.squeeze()

        edge_feat = self.edge_nn(edge_attr)
        edge_feat = edge_feat.unsqueeze(1)
        order2 = self.order2(torch.relu(u1_i + u2_j + edge_feat))
        u_j = u_j + order2
        return u_j

    def update(self, aggr_u, u):
        up1 = self.update1(torch.cat((u, aggr_u), dim=-1))
        up2 = up1 + self.update2(up1)
        return up2 + u
