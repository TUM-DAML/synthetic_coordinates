import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from .torch_vertex import GENConv_Linegraph
from .torch_nn import norm_layer


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
        block="res+",
        hidden_channels=256,
        conv="gen",
        gcn_aggr="softmax",
        learn_t=True,
        t=1.0,
        learn_p=False,
        p=1,
        msg_norm=False,
        learn_msg_scale=False,
        norm="batch",
        mlp_layers=1,
        graph_pooling="mean",
        node_attr_dim=None,
        edge_attr_dim=None,
        mlp_act="relu",
        lg_node_basis=4,
        lg_edge_basis=4,
        emb_basis_global=True,
        emb_basis_local=False,
        emb_bottleneck=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.node_features_encoder = torch.nn.Linear(node_attr_dim, hidden_channels)

        # layer to embed the LG node message
        # 2 node embeddings + edge feature + dist basis dimension
        self.msg_emb_layer = nn.Linear(
            2 * hidden_channels + edge_attr_dim + lg_node_basis, hidden_channels
        )

        self.pool = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool,
        }[graph_pooling]

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

        self.block = block
        if self.block == "res+":
            print("LN/BN->ReLU->GraphConv->Res")
        else:
            raise Exception("Unknown block Type")

        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.conv = conv
        self.num_layers = num_layers

        for _ in range(num_layers):
            if conv == "gen":
                gcn = GENConv_Linegraph(
                    hidden_channels,
                    hidden_channels,
                    aggr=gcn_aggr,
                    t=t,
                    learn_t=learn_t,
                    p=p,
                    learn_p=learn_p,
                    msg_norm=msg_norm,
                    learn_msg_scale=learn_msg_scale,
                    edge_attr_dim=edge_attr_dim,
                    norm=norm,
                    mlp_layers=mlp_layers,
                    mlp_act=mlp_act,
                    lg_node_basis=lg_node_basis,
                    lg_edge_basis=lg_edge_basis,
                    emb_basis_global=emb_basis_global,
                    emb_basis_local=emb_basis_local,
                    emb_bottleneck=emb_bottleneck,
                )
            else:
                raise Exception("Unknown Conv Type")
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

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

        if self.block == "res+":
            if self.conv == "gen":
                h = self.gcns[0](h, edge_index, node_basis, edge_basis)

            for layer in range(1, self.num_layers):
                # norm-relu-dropout-GCN
                # batch/instance/layer norm
                h1 = self.norms[layer - 1](h)
                # relu before GCN, so that layer output can be negative
                h2 = F.relu(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.conv == "gen":
                    h = self.gcns[layer](h2, edge_index, node_basis, edge_basis) + h

            # last norm layer
            h = self.norms[self.num_layers - 1](h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        final_node_emb = torch_scatter.scatter_add(
            h, batch.edge_index_g[1], dim=0, dim_size=batch.x_g.shape[0]
        )  # this is graph.num_nodes

        h_graph = self.pool(final_node_emb, batch.batch)

        return self.graph_pred_linear(h_graph)
