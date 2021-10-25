import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import DNAConv, JumpingKnowledge, GATConv
import torch_geometric
import torch_scatter

import numpy as np

from .message_layer import MessageLayer


class DimeNetEmbedding(nn.Module):
    """docstring for DimeNetFeatureEmbedding"""
    def __init__(self, g_feat_dim, g_edgeattr_dim, emb_dim, dropout=0.5):
        super(DimeNetEmbedding, self).__init__()
        self.dropout = dropout

        self.feat_emb_layer = nn.Linear(g_feat_dim, emb_dim)
        self.msg_emb_layer = nn.Linear(2 * emb_dim + g_edgeattr_dim, emb_dim)
        self.output = DimeNetOutput(g_edgeattr_dim, emb_dim, dropout)

    def reset_parameters(self):
        self.feat_emb_layer.reset_parameters()
        self.msg_emb_layer.reset_parameters()

    def forward(self, graph):
        vertex_feat = F.relu(F.dropout(self.feat_emb_layer(graph.x), self.dropout, training=self.training))

        # Generate message embeddings for edge (u, v)
        # feature(u) for every edge (u, v)
        msg_emb_u = torch.index_select(vertex_feat, dim=0, index=graph.edge_index[0])
        # feature(v) for every edge (u, v)
        msg_emb_v = torch.index_select(vertex_feat, dim=0, index=graph.edge_index[1])
        # concat both features with graph.edge_attr
        msg = F.relu(torch.cat((msg_emb_u, msg_emb_v, graph.edge_attr), dim=-1))

        msg_emb = self.msg_emb_layer(msg)

        return vertex_feat, msg_emb

class DimeNetResidual(nn.Module):
    """docstring for DimeNetResidual"""
    def __init__(self, dim):
        super(DimeNetResidual, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x2 = F.relu(self.linear2(x1))

        return x + x2

class DimeNetOutput(nn.Module):
    """docstring for DimeNetOutput"""
    def __init__(self, g_edgeattr_dim, emb_dim, dropout=0.5):
        super(DimeNetOutput, self).__init__()
        self.dropout = dropout

        self.edgeattr_linear = nn.Linear(g_edgeattr_dim, emb_dim)

        self.linears = nn.ModuleList(
            [nn.Linear(emb_dim, emb_dim) for _ in range(3)]
            )

        self.linear4 = nn.Linear(emb_dim, emb_dim, bias=False)

    def reset_parameters(self):
        self.edgeattr_linear.reset_parameters()
        for l in self.linears:
            l.reset_parameters()
        self.linear4.reset_parameters()

    def forward(self, graph, msg_emb):
        edgeattr_emb = self.edgeattr_linear(graph.edge_attr)
        x = edgeattr_emb * msg_emb

        node_emb = torch_scatter.scatter_add(x, graph.edge_index[1], dim=0, dim_size=graph.num_nodes)

        x1 = F.relu(self.linears[0](node_emb))
        x2 = F.relu(self.linears[0](x1))
        x3 = F.relu(self.linears[0](x2))

        return F.dropout(self.linear4(x3), self.dropout, training=self.training)

class DimeNetInteraction(nn.Module):
    """docstring for DimeNetInteraction"""
    def __init__(self, g_edgeattr_dim, emb_dim, lg_edgeattr_dim, dropout):
        super(DimeNetInteraction, self).__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim)

        self.linegraph_msg_layer = MessageLayer(emb_dim, lg_edgeattr_dim,
                                                F.relu, 0.5, bias=False)
        self.residuals = nn.ModuleList(
            [DimeNetResidual(emb_dim) for _ in range(3)]
            )

        self.linear2 = nn.Linear(emb_dim, emb_dim)

        self.output = DimeNetOutput(g_edgeattr_dim, emb_dim, dropout)

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linegraph_msg_layer.reset_parameters()
        for l in self.residuals:
            l.reset_parameters()
        self.linear2.reset_parameters()
        self.output.reset_parameters()

    def forward(self, msg_emb, graph, linegraph):
        prev_msg_new_emb = F.relu(self.linear1(msg_emb))
        new_msg_emb = self.linegraph_msg_layer(msg_emb, linegraph.edge_index, linegraph.edge_attr)

        x = self.residuals[0](prev_msg_new_emb + new_msg_emb)
        x1 = F.relu(self.linear2(x))
        x3 = self.residuals[1](x1)
        x4 = self.residuals[2](x3)

        x5 = self.output(graph, x4)

        return x4, x5


class ICDimeNet(nn.Module):
    def __init__(self, g_x_dim, num_classes, g_edgeattr_dim, lg_edgeattr_dim,
                 emb_dim=64, num_msg_layers=1, dropout=0.5):
        '''
        g_x_dim: dim of graph.x
        num_classes: number of output classes
        g_edgeattr_dim: dim of graph.edge_attr
        lg_edgeattr_dim: dim of linegraph.edge_attr

        emb_size: dim of the embedding of each vertex
        num_msg_layers: number of message passing layers

        dropout: prob of dropping every intermediate output
        '''
        super(ICDimeNet, self).__init__()
        self.dropout = dropout

        self.input_emb = DimeNetEmbedding(g_x_dim, g_edgeattr_dim, emb_dim, dropout)

        self.interactions = nn.ModuleList(
                [DimeNetInteraction(g_edgeattr_dim, emb_dim, lg_edgeattr_dim, dropout) \
                    for _ in range(num_msg_layers)]
            )
        self.aggr = nn.Parameter(torch.ones(num_msg_layers + 1, 1, 1))
        self.out_layer = nn.Linear(emb_dim, num_classes)

    def reset_parameters(self):
        self.input_emb.reset_parameters()
        for l in self.interactions:
            l.reset_parameters()
        self.out_layer.reset_parameters()

    def forward(self, graph, linegraph):
        outputs = []

        vertex_feat, msg_emb = self.input_emb(graph)
        outputs.append(vertex_feat)

        for layer_ndx, i_layer in enumerate(self.interactions):
            msg_emb, output = i_layer(msg_emb, graph, linegraph)
            outputs.append(output)

        outputs = torch.stack(outputs)
        weights = F.softmax(self.aggr, dim=0)
        aggr_output = torch.sum(weights * outputs, dim=0)
        return F.log_softmax(self.out_layer(aggr_output), dim=-1)
