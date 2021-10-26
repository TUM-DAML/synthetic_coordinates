import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import DNAConv, JumpingKnowledge, GATConv
import torch_geometric
import torch_scatter

import numpy as np


class JKGATConvNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(JKGATConvNet, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, 8, heads=8, concat=True, dropout=0.6)

        self.out_layer = nn.Linear(64, num_classes, bias=False)

        self.jk = JumpingKnowledge("lstm", 64, 2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.out_layer.reset_parameters()

    def forward(self, x, edge_index):
        x1 = F.dropout(x, p=0.6, training=self.training)
        x1 = F.elu(self.conv1(x1, edge_index))

        x2 = F.dropout(x1, p=0.6, training=self.training)
        x2 = self.conv2(x2, edge_index)

        final_emb = self.jk([x1, x2])
        # use only final layer prediction
        return F.log_softmax(self.out_layer(final_emb), dim=-1)


class GATConvNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GATConvNet, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_classes, heads=8, concat=True, dropout=0.6)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
