import torch
from ogb.graphproppred.mol_encoder import BondEncoder
from .torch_nn import MLP
from .torch_message import GenMessagePassing, MsgNorm

from torch.nn import Linear, Sequential, Identity


class GENConv_Linegraph(GenMessagePassing):
    """
    GENeralized Graph Convolution (GENConv): https://arxiv.org/pdf/2006.07739.pdf
    SoftMax  &  PowerMean Aggregation
    """

    def __init__(
        self,
        in_dim,
        emb_dim,
        aggr="softmax",
        t=1.0,
        learn_t=False,
        p=1.0,
        learn_p=False,
        msg_norm=False,
        learn_msg_scale=True,
        norm="batch",
        mlp_layers=2,
        mlp_act="relu",
        lg_node_basis=None,
        lg_edge_basis=None,
        emb_basis_global=True,
        emb_basis_local=False,
        emb_bottleneck=False,
    ):

        super().__init__(aggr=aggr, t=t, learn_t=learn_t, p=p, learn_p=learn_p)

        channels_list = [in_dim]

        for _ in range(mlp_layers - 1):
            channels_list.append(in_dim * 2)

        channels_list.append(emb_dim)

        self.mlp = MLP(channels=channels_list, norm=norm, last_lin=False, act=mlp_act)

        if emb_basis_local:
            # has the basis been embedded to the bottleneck globally?
            if emb_basis_global:
                # then do bottleneck->hidden
                self.emb_node_basis = Linear(emb_bottleneck, emb_dim)
                self.emb_edge_basis = Linear(emb_bottleneck, emb_dim)
            else:
                # need to embed the basis here
                if emb_bottleneck:
                    # basis->bottleneck->hidden
                    self.emb_node_basis = Sequential(
                        Linear(lg_node_basis, emb_bottleneck),
                        Linear(emb_bottleneck, emb_dim),
                    )
                    self.emb_edge_basis = Sequential(
                        Linear(lg_edge_basis, emb_bottleneck),
                        Linear(emb_bottleneck, emb_dim),
                    )
                else:
                    # basis->hidden
                    self.emb_node_basis = Linear(lg_node_basis, emb_dim)
                    self.emb_edge_basis = Linear(lg_edge_basis, emb_dim)
        else:
            # basis already embedded to emb_dim
            self.emb_node_basis = Identity()
            self.emb_edge_basis = Identity()

        # Linear layers similar to DimeNet
        self.msg_emb1 = Linear(emb_dim, emb_dim)
        self.msg_emb2 = Linear(emb_dim, emb_dim, bias=False)
        self.msg_emb3 = Linear(emb_dim, emb_dim, bias=False)

        self.msg_norm = (
            MsgNorm(learn_msg_scale=learn_msg_scale) if msg_norm else Identity()
        )

    def forward(self, x, edge_index, node_basis, edge_basis):
        """
        Pass the whole linegraph through the layer
        x: input embedding on nodes
        edge_index
        node_basis: feature for each node
        edge_basis: feature for each edge
        """
        # start the message passing, get messages for each node
        m = self.propagate(
            edge_index, x=x, node_basis=node_basis, edge_basis=edge_basis
        )
        m = self.msg_norm(m)
        # skip connection
        h = x + m
        # embed with MLP
        out = self.mlp(h)

        return out

    def message(self, x_j, node_basis_i, edge_basis):
        """
        Message passing: Dimenet interaction layer

        at a node i, for each neighbor j
        x_j: the embedding at the neighbor
        node_basis_i: basis vector at the current node
        edge_basis:
        """
        msg1 = x_j * self.emb_node_basis(node_basis_i)
        msg2 = self.msg_emb2(msg1)
        msg3 = msg2 * self.emb_edge_basis(edge_basis)
        msg4 = self.msg_emb3(msg3)

        return msg4

    def update(self, aggr_out):
        return aggr_out


class GENConv(GenMessagePassing):
    """
    GENeralized Graph Convolution (GENConv): https://arxiv.org/pdf/2006.07739.pdf
    SoftMax  &  PowerMean Aggregation
    """

    def __init__(
        self,
        in_dim,
        emb_dim,
        aggr="softmax",
        t=1.0,
        learn_t=False,
        p=1.0,
        learn_p=False,
        msg_norm=False,
        learn_msg_scale=True,
        edge_feat_dim=None,
        norm="batch",
        mlp_layers=2,
        mlp_act="relu",
        emb_basis_global=True,
        emb_basis_local=True,
        emb_bottleneck=4,
    ):

        super(GENConv, self).__init__(
            aggr=aggr, t=t, learn_t=learn_t, p=p, learn_p=learn_p
        )

        channels_list = [in_dim]

        for _ in range(mlp_layers - 1):
            channels_list.append(in_dim * 2)

        channels_list.append(emb_dim)

        self.mlp = MLP(channels=channels_list, norm=norm, last_lin=False, act=mlp_act)

        if emb_basis_local:
            # has the basis been embedded to the bottleneck globally?
            if emb_basis_global:
                # then do bottleneck->hidden
                self.emb_edge_attr = Linear(emb_bottleneck, emb_dim)
            else:
                # need to embed the basis here
                if emb_bottleneck:
                    # basis->bottleneck->hidden
                    self.emb_edge_attr = Sequential(
                        Linear(edge_feat_dim, emb_bottleneck),
                        Linear(emb_bottleneck, emb_dim),
                    )
                else:
                    # basis->hidden
                    self.emb_edge_attr = Linear(edge_feat_dim, emb_dim)
        else:
            # basis already embedded to emb_dim
            self.emb_edge_attr = Identity()

        # Linear layers similar to DimeNet
        self.msg_emb1 = Linear(emb_dim, emb_dim)
        self.msg_emb2 = Linear(emb_dim, emb_dim, bias=False)

        self.msg_norm = (
            MsgNorm(learn_msg_scale=learn_msg_scale) if msg_norm else Identity()
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: input node embedding at this layer
        edge_index: usual
        edge_attr: usual (may include the distance basis)
        """
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        m = self.msg_norm(m)
        h = x + m
        out = self.mlp(h)

        return out

    def message(self, x_j, edge_attr):
        msg1 = self.msg_emb1(x_j) * self.emb_edge_attr(edge_attr)
        msg2 = self.msg_emb2(msg1)

        return msg2

    def update(self, aggr_out):
        return aggr_out
