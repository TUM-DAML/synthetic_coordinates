from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin


class MLP(Seq):
    def __init__(self, channels):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], True))
            m.append(nn.BatchNorm1d(channels[i], affine=True))
            m.append(nn.ReLU())

        self.m = m
        super(MLP, self).__init__(*self.m)
