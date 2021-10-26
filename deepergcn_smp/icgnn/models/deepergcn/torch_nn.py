import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential as Seq, Linear as Lin


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


##############################
#    Basic layers
##############################
def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act_type.lower()
    if act == "relu":
        layer = nn.ReLU(inplace)
    elif act == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == "swish":
        layer = Swish()
    else:
        raise NotImplementedError("activation layer [%s] is not found" % act)
    return layer


def norm_layer(norm_type, nc):
    # normalization layer 1d
    norm = norm_type.lower()
    if norm == "batch":
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == "layer":
        layer = nn.LayerNorm(nc, elementwise_affine=True)
    elif norm == "instance":
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm)
    return layer


class MLP(Seq):
    def __init__(
        self, channels, act="relu", norm=None, bias=True, drop=0.0, last_lin=False
    ):
        m = []
        for i in range(1, len(channels)):

            m.append(Lin(channels[i - 1], channels[i], bias))

            if (i == len(channels) - 1) and last_lin:
                pass
            else:
                if norm:
                    m.append(norm_layer(norm, channels[i]))
                if act:
                    m.append(act_layer(act))
                if drop > 0:
                    m.append(nn.Dropout2d(drop))

        self.m = m
        super(MLP, self).__init__(*self.m)
