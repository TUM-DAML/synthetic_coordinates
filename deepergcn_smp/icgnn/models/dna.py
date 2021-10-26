import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import DNAConv, JumpingKnowledge, GATConv
import torch_geometric
import torch_scatter

import numpy as np


class DNAConvNet(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, heads=1, groups=1
    ):
        super(DNAConvNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                DNAConv(hidden_channels, heads, groups, dropout=0.8, cached=True)
            )
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.log_softmax(x, dim=1)
