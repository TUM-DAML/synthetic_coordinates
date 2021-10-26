import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import JumpingKnowledge, BatchNorm
import torch_geometric
import torch_scatter
from torch_geometric.nn import global_mean_pool as gap

import numpy as np

from .message_layer import MessageLayer


class ICGNN(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_dist_basis,
        num_cos_basis,
        degree_norm,
        msgpass_aggr,
        batch_norm=False,
        emb_size=32,
        num_msg_layers=1,
        dropout=0.5,
        act_fn=F.relu,
        lg_msg_dim=32,
        graphcls=False,
        eval_ogb=False,
    ):
        """
        num_features: the dim of the feature of each vertex
        num_classes: number of output classes for each vertex
        num_dist_basis: the number of features representing the edge of the graph
        num_cos_basis: the number of features representing angle between adjacent edges
        emb_size: dim of the embedding of each vertex
        num_msg_layers: number of message passing layers
        dropout: prob of dropping every intermediate output
        act_fn: activation function
        lg_msg_dim: dimension of the message in the linegraph
        graphcls: graph classification? else node classification
        eval_ogb: using OGB dataset? (output logits, not logsoftmax)
        """
        super().__init__()
        # True if graph classification, else node classification
        self.graphcls = graphcls
        self.act_fn = act_fn
        self.dropout = nn.Dropout(dropout)

        self.eval_ogb = eval_ogb
        self.num_classes = num_classes

        # layers that embed the graph inputs
        # embed the node feature
        self.feat_emb_layer = nn.Linear(num_features, emb_size, bias=False)
        # one-time creation of LG features using graph features
        self.msg_emb_layer = nn.Linear(
            2 * emb_size + num_dist_basis, lg_msg_dim, bias=False
        )
        # convert lgraph msg to the same size as graph msg
        self.lg_to_g = nn.Linear(lg_msg_dim, emb_size)

        self.batch_norm = batch_norm

        # forward prop layers ("message passing")
        msg_layers = []

        self.bn_layers = []

        for _ in range(num_msg_layers):
            msg_layers.append(
                MessageLayer(
                    lg_msg_dim,
                    num_cos_basis,
                    self.act_fn,
                    dropout,
                    aggr=msgpass_aggr,
                    degree_norm=degree_norm,
                    bias=False,
                )
            )
            # add batch norm after every message passing layer
            if batch_norm:
                self.bn_layers.append(BatchNorm(lg_msg_dim))

        if self.bn_layers:
            self.bn_layers = nn.ModuleList(self.bn_layers)

        self.msg_layers = nn.ModuleList(msg_layers)

        self.jk = JumpingKnowledge("lstm", emb_size, num_msg_layers + 2)

        self.out_layer = nn.Linear(emb_size, self.num_classes, bias=False)

    def reset_parameters(self):
        for layer in self.msg_layers:
            layer.reset_parameters()

        self.feat_emb_layer.reset_parameters()
        self.msg_emb_layer.reset_parameters()
        self.out_layer.reset_parameters()

    def forward(self, graph=None, linegraph=None, batch=None):
        # check if graphcls or node classification
        if self.graphcls:
            graph = Data(
                x=batch.x_g, edge_index=batch.edge_index_g, edge_attr=batch.edge_attr_g
            )
            linegraph = Data(
                x=batch.x_lg,
                edge_index=batch.edge_index_lg,
                edge_attr=batch.edge_attr_lg,
            )
            batch_ndx = batch.x_g_batch

        # Embed each node independently - layer1 embedding
        # one for each node
        # if the input is not a float tensor (eg: LongTensor), convert it
        x = graph.x.to(torch.float32)
        node_emb = self.act_fn(self.dropout(self.feat_emb_layer(x)))

        # Generate message embeddings for edge (u, v)
        # feature(u) for every edge (u, v)
        msg_emb1 = torch.index_select(node_emb, dim=0, index=graph.edge_index[0])
        # feature(v) for every edge (u, v)
        msg_emb2 = torch.index_select(node_emb, dim=0, index=graph.edge_index[1])
        # concat both features with edge feature(u, v) which is the L2 distance
        msg_emb_cat = torch.cat((msg_emb1, msg_emb2, graph.edge_attr), dim=-1)
        # create the embedding for this message
        msg_emb = self.act_fn(self.dropout(self.msg_emb_layer(msg_emb_cat)))
        msg_emb_reduced = self.lg_to_g(msg_emb)

        # do this before every append: get one vector for each graph
        if self.graphcls:
            node_emb = gap(node_emb, batch_ndx)
        node_embs = [node_emb]

        # add the embedded incoming message at each vertex to the vertex embedding
        node_emb = torch_scatter.scatter_add(
            msg_emb_reduced, graph.edge_index[1], dim=0, dim_size=graph.num_nodes
        )

        # do this before every append: get one vector for each graph
        if self.graphcls:
            node_emb = gap(node_emb, batch_ndx)
        node_embs.append(node_emb)

        # Propagate and transform message embeddings
        for (ndx, layer) in enumerate(self.msg_layers):
            msg_emb = layer(msg_emb, linegraph.edge_index, linegraph.edge_attr)
            # scatter_add aggregates all the messages going *to* each destination vertex
            # result is one vector per node
            incoming_msg = torch_scatter.scatter_add(
                msg_emb, graph.edge_index[1], dim=0, dim_size=graph.num_nodes
            )

            # apply batch norm
            if self.bn_layers:
                incoming_msg = self.bn_layers[ndx](incoming_msg)

            incoming_msg_reduced = self.lg_to_g(incoming_msg)
            # append this layer n embedding (1 for each vertex) to the same list
            if self.graphcls:
                incoming_msg_reduced = gap(incoming_msg_reduced, batch_ndx)
            node_embs.append(incoming_msg_reduced)

        # get the final embedding by applying the jumping knowledge
        # on all the embeddings collected till now
        final_emb = self.jk(node_embs)
        # use only final layer prediction

        out = self.out_layer(final_emb)
        # binary classification - output a single logit for each sample
        if self.num_classes == 1:
            return out
        # multi class - output log_softmax
        else:
            return F.log_softmax(out, dim=-1)
