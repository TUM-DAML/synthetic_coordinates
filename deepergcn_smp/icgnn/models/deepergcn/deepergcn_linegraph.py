import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .torch_vertex import GENConv_Linegraph


class DeeperGCN_LineGraph(torch.nn.Module):
    """
    DeeperGCN on the Linegraph

    Takes graph+linegraph as input
    Prepares the first message embedding using graph.x, graph.edge_attr
    Then does message passing on the linegraph using lg.x, lg.edge_attr
    Add incoming messages to get back graph node embeddings
    """

    def __init__(
        self,
        num_tasks=None,
        num_layers=7,
        dropout=0.2,
        hidden_channels=256,
        node_attr_dim=None,
        edge_attr_dim=None,
        lg_node_basis=4,
        lg_edge_basis=4,
        # options for distance and angle basis
        emb_basis_global=True,
        emb_basis_local=True,
        emb_bottleneck=4,
    ):
        super().__init__()
        self.dropout = dropout
        self.node_features_encoder = torch.nn.Linear(node_attr_dim, hidden_channels)

        # layer to embed the LG node message
        # 2 node embeddings + edge feature + dist basis dimension
        self.msg_emb_layer = nn.Linear(
            2 * hidden_channels + edge_attr_dim + lg_node_basis, hidden_channels
        )

        # nodes and edges have extra basis = distances, angles
        if emb_basis_global:
            if emb_bottleneck:
                if emb_basis_local:
                    # basis->emb_bottleneck
                    self.global_emb_node_basis = nn.Linear(
                        lg_node_basis, emb_bottleneck
                    )
                    self.global_emb_edge_basis = nn.Linear(
                        lg_edge_basis, emb_bottleneck
                    )
                else:
                    # basis->bottleneck->hidden_channels
                    self.global_emb_node_basis = nn.Sequential(
                        nn.Linear(lg_node_basis, emb_bottleneck),
                        nn.Linear(emb_bottleneck, hidden_channels),
                    )
                    self.global_emb_edge_basis = nn.Sequential(
                        nn.Linear(lg_edge_basis, emb_bottleneck),
                        nn.Linear(emb_bottleneck, hidden_channels),
                    )
            else:
                # basis->hidden_channels
                self.global_emb_node_basis = nn.Linear(lg_node_basis, hidden_channels)
                self.global_emb_edge_basis = nn.Linear(lg_edge_basis, hidden_channels)
        else:
            self.global_emb_node_basis = nn.Identity()
            self.global_emb_edge_basis = nn.Identity()

        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.num_layers = num_layers

        for _ in range(num_layers):
            gcn = GENConv_Linegraph(
                hidden_channels,
                hidden_channels,
                lg_node_basis=lg_node_basis,
                lg_edge_basis=lg_edge_basis,
                emb_basis_global=emb_basis_global,
                emb_basis_local=emb_basis_local,
                emb_bottleneck=emb_bottleneck,
            )
            self.gcns.append(gcn)
            self.norms.append(nn.BatchNorm1d(hidden_channels, affine=True))

        # final prediction layer
        self.graph_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, batch):
        # initial embedding of graph node feature
        h = self.node_features_encoder(batch.x_g.to(torch.float32))

        # construct the first message = h_uv || h_vu || edgeattr_uv
        msg1 = torch.index_select(h, dim=0, index=batch.edge_index_g[0])
        msg2 = torch.index_select(h, dim=0, index=batch.edge_index_g[1])
        msg_concat = torch.cat(
            (msg1, msg2, batch.edge_attr_g.to(torch.float32), batch.x_lg), dim=-1
        )
        h = self.msg_emb_layer(msg_concat)

        edge_index = batch.edge_index_lg
        node_basis = self.global_emb_node_basis(batch.edge_dist_basis)
        edge_basis = self.global_emb_edge_basis(batch.edge_attr_lg)

        h = self.gcns[0](h, edge_index, node_basis, edge_basis)

        for layer in range(1, self.num_layers):
            # norm-relu-dropout-GCN
            # batch/instance/layer norm
            h1 = self.norms[layer - 1](h)
            # relu before GCN, so that layer output can be negative
            h2 = F.relu(h1)
            h2 = F.dropout(h2, p=self.dropout, training=self.training)

            # local embedding is inside the GCN layer
            h = self.gcns[layer](h2, edge_index, node_basis, edge_basis) + h

        # last norm layer
        h = self.norms[self.num_layers - 1](h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        final_node_emb = torch_scatter.scatter_add(
            h, batch.edge_index_g[1], dim=0, dim_size=batch.x_g.shape[0]
        )  # this is graph.num_nodes

        h_graph = global_mean_pool(final_node_emb, batch.batch)

        return self.graph_pred_linear(h_graph)
