import logging

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import PNAConv

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from .torch_vertex import GENConv
from .torch_nn import norm_layer, MLP


class DeeperGCN(torch.nn.Module):
    def __init__(
        self,
        num_tasks,
        num_layers=7,
        dropout=0.2,
        block="res+",
        conv_encode_edge=False,
        add_virtual_node=False,
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
        edge_feat_dim=None,
        mol_data=True,
        code_data=False,
        node_encoder=None,
        node_feat_dim=7,
        emb_product=True,
        mlp_act="relu",
        emb_use_both=False,
        deg_hist=None,
        emb_bottleneck=False,
        emb_use_global=True,
    ):
        super(DeeperGCN, self).__init__()

        self.conv = conv
        self.num_layers = num_layers
        self.dropout = dropout
        self.block = block
        self.conv_encode_edge = conv_encode_edge
        self.add_virtual_node = add_virtual_node
        # molhiv and molpcba
        self.mol_data = mol_data
        # for ogbg-code
        self.code_data = code_data
        self.node_encoder = node_encoder
        self.num_tasks = num_tasks
        self.emb_use_global = emb_use_global

        # use one global embedding for node and edge attributes
        if self.emb_use_global:
            self.linear_edgeattr = Linear(edge_feat_dim, hidden_channels)

        self.learn_t = learn_t
        self.learn_p = learn_p
        self.msg_norm = msg_norm

        if not self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)
            edge_feat_dim = hidden_channels
            print(">> Override edge_feat_dim with hidden_channels")

        print(
            "The number of layers {}".format(self.num_layers),
            "Aggr aggregation method {}".format(gcn_aggr),
            "block: {}".format(self.block),
        )
        if self.block == "res+":
            print("LN/BN->ReLU->GraphConv->Res")
        elif self.block == "res":
            print("GraphConv->LN/BN->ReLU->Res")
        elif self.block == "dense":
            raise NotImplementedError("To be implemented")
        elif self.block == "plain":
            print("GraphConv->LN/BN->ReLU")
        else:
            raise Exception("Unknown block Type")

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels] * 3, norm=norm))

        for layer in range(self.num_layers):
            if conv == "pna":
                gcn = PNAConv(
                    hidden_channels,
                    hidden_channels,
                    aggregators=["max", "min", "mean", "std"],
                    scalers=["identity", "amplification", "attenuation"],
                    deg=deg_hist,
                    edge_dim=edge_feat_dim,
                )
            elif conv == "gen":
                gcn = GENConv(
                    hidden_channels,
                    hidden_channels,
                    aggr=gcn_aggr,
                    t=t,
                    learn_t=self.learn_t,
                    p=p,
                    learn_p=self.learn_p,
                    msg_norm=self.msg_norm,
                    learn_msg_scale=learn_msg_scale,
                    encode_edge=self.conv_encode_edge and not self.emb_use_global,
                    bond_encoder=False,
                    edge_feat_dim=edge_feat_dim,
                    norm=norm,
                    mlp_layers=mlp_layers,
                    emb_product=emb_product,
                    mlp_act=mlp_act,
                    # if no global embedding
                    # embed the attrs in each layer
                    emb_attrs=not self.emb_use_global,
                    emb_use_both=emb_use_both,
                    emb_bottleneck=emb_bottleneck,
                )
            else:
                raise Exception("Unknown Conv Type")
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        # molecule data?
        if mol_data:
            self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)
        else:
            self.node_features_encoder = torch.nn.Linear(node_feat_dim, hidden_channels)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception("Unknown Pool Type")

        if self.code_data:
            # predict 5 tokens of the function name for each graph
            self.graph_pred_linear_list = torch.nn.ModuleList()
            for i in range(self.num_tasks):
                self.graph_pred_linear_list.append(
                    torch.nn.Linear(hidden_channels, 5000 + 2)
                )
        else:
            self.graph_pred_linear = torch.nn.Linear(hidden_channels, self.num_tasks)

    def forward(self, input_batch, h=None, make_pred=True):
        """
        input_batch: the Batch of data objects
        h: the initial embedding, created outside. Otherwise create it here
        make_pred: make predictions? otherwise return last layer activations
        """
        x = input_batch.x
        edge_index = input_batch.edge_index
        edge_attr = input_batch.edge_attr
        batch = input_batch.batch

        x_emb = None

        # check if initial embedding is already provided
        if h is None:
            if self.mol_data:
                h = self.atom_encoder(x)
            elif self.code_data:
                node_depth = input_batch.node_depth
                h = self.node_encoder(
                    x,
                    node_depth.view(
                        -1,
                    ),
                )
            else:
                h = self.node_features_encoder(x.float())
        else:
            x_emb = self.node_features_encoder(x)
            h = h + x_emb

        if self.add_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1)
                .to(edge_index.dtype)
                .to(edge_index.device)
            )
            h = h + virtualnode_embedding[batch]

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        if self.emb_use_global:
            edge_emb = self.linear_edgeattr(edge_attr.float())

        if self.block == "res+":
            # get the initial vertex features
            # only GCN!
            if self.conv == "gen":
                h = self.gcns[0](h, edge_index, edge_emb, x_emb)
            else:
                h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                # norm-relu-dropout-GCN
                # batch/instance/layer norm
                h1 = self.norms[layer - 1](h)
                # relu before GCN, so that layer output can be negative
                h2 = F.relu(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.add_virtual_node:
                    virtualnode_embedding_temp = (
                        global_add_pool(h2, batch) + virtualnode_embedding
                    )
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer - 1](
                            virtualnode_embedding_temp
                        ),
                        self.dropout,
                        training=self.training,
                    )

                    h2 = h2 + virtualnode_embedding[batch]
                # the actual GCN = GENConv
                if self.conv == "gen":
                    h = self.gcns[layer](h2, edge_index, edge_emb, x_emb) + h
                else:
                    h = self.gcns[layer](h2, edge_index, edge_emb) + h

            # last norm layer
            h = self.norms[self.num_layers - 1](h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == "res":
            # GCN-norm-relu-dropout on the input
            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                # GCN-norm-relu-dropout
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == "plain":

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = F.relu(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception("Unknown block Type")

        if make_pred:
            h_graph = self.pool(h, batch)

            if self.code_data:
                pred_list = []

                for linear in self.graph_pred_linear_list:
                    pred_list.append(linear(h_graph))

                return pred_list
            else:
                return self.graph_pred_linear(h_graph)
        else:
            return h

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print("Final t {}".format(ts))
            else:
                logging.info("Epoch {}, t {}".format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print("Final p {}".format(ps))
            else:
                logging.info("Epoch {}, p {}".format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print("Final s {}".format(ss))
            else:
                logging.info("Epoch {}, s {}".format(epoch, ss))
