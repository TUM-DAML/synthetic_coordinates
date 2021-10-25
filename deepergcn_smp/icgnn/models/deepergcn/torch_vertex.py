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
    def __init__(self, in_dim, emb_dim,
                aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False,
                 msg_norm=False, learn_msg_scale=True,
                 edge_attr_dim=None,
                 norm='batch', mlp_layers=2, mlp_act='relu', 
                 lg_node_basis=None, lg_edge_basis=None,
                 emb_basis_global=True,
                 emb_basis_local=False, 
                 emb_bottleneck=False):

        super().__init__(aggr=aggr,
                            t=t, learn_t=learn_t,
                            p=p, learn_p=learn_p)

        channels_list = [in_dim]

        for _ in range(mlp_layers-1):
            channels_list.append(in_dim*2)

        channels_list.append(emb_dim)

        self.mlp = MLP(channels=channels_list,
                       norm=norm,
                       last_lin=False,
                       act=mlp_act)

        self.msg_encoder = torch.nn.ReLU()
        self.eps = 1e-7

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
                        Linear(emb_bottleneck, emb_dim)
                    )
                    self.emb_edge_basis = Sequential(
                        Linear(lg_edge_basis, emb_bottleneck), 
                        Linear(emb_bottleneck, emb_dim)
                    )
                else:
                    # basis->hidden
                    self.emb_node_basis = Linear(lg_node_basis, emb_dim)
                    self.emb_edge_basis = Linear(lg_edge_basis, emb_dim)
        else:
            # basis already embedded to emb_dim
            self.emb_node_basis = Identity()
            self.emb_edge_basis = Identity()
        
        # embed the message a few times
        self.msg_emb1 = Linear(emb_dim, emb_dim)
        self.msg_emb2 = Linear(emb_dim, emb_dim, bias=False)
        self.msg_emb3 = Linear(emb_dim, emb_dim, bias=False)

        if msg_norm:
            self.msg_norm = MsgNorm(learn_msg_scale=learn_msg_scale)
        else:
            self.msg_norm = Identity()

    def forward(self, x, edge_index, node_basis, edge_basis):
        m = self.propagate(edge_index, x=x, node_basis=node_basis, edge_basis=edge_basis)
        m = self.msg_norm(m)
        # skip connection
        h = x + m
        # embed with MLP
        out = self.mlp(h)

        return out

    def message(self, x_j, node_basis_i, edge_basis):
        '''
        Message passing: Dimenet interaction layer
        '''
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
    def __init__(self, in_dim, emb_dim,
                 aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False,
                 msg_norm=False, learn_msg_scale=True,
                 encode_edge=False, bond_encoder=False,
                 edge_feat_dim=None,
                 norm='batch', mlp_layers=2,
                 eps=1e-7, emb_product=True,
                 mlp_act='relu', 
                 emb_attrs=True,
                 emb_use_both=False,
                 emb_bottleneck=False):

        super(GENConv, self).__init__(aggr=aggr,
                                      t=t, learn_t=learn_t,
                                      p=p, learn_p=learn_p)

        channels_list = [in_dim]


        for _ in range(mlp_layers-1):
            channels_list.append(in_dim*2)

        channels_list.append(emb_dim)

        self.mlp = MLP(channels=channels_list,
                       norm=norm,
                       last_lin=False,
                       act=mlp_act)

        self.msg_encoder = torch.nn.ReLU()
        self.eps = eps
        
        self.emb_attrs = emb_attrs
        self.emb_product = emb_product
        self.emb_use_both = emb_use_both
        # extra layers to implement Dimenet architecture

        if self.emb_product and self.emb_use_both:
            # project the (distance_embedding*x) once before multiplying with
            # angle embedding
            self.linear_intermediate = Linear(emb_dim, emb_dim, bias=False)
            # need to emb the attributes, or do we get the global embedding?
            if self.emb_attrs:
                if emb_bottleneck:
                    # use 2 linear layers to project
                    # d -> bottleneck -> d
                    self.linear_target_x = Sequential(
                        Linear(emb_dim, emb_bottleneck, bias=False),
                        Linear(emb_bottleneck, emb_dim, bias=False)
                    )
                    self.linear_edgeattr = Sequential(
                        Linear(emb_dim, emb_bottleneck, bias=False),
                        Linear(emb_bottleneck, emb_dim, bias=False)
                    )
                else:
                    # use 1 linear layer to project, d -> d
                    self.linear_target_x = Linear(emb_dim, emb_dim, bias=False)
                    self.linear_edgeattr = Linear(emb_dim, emb_dim, bias=False)
            else:
                self.linear_target_x = Identity()
                self.linear_edgeattr = Identity()                    

        self.msg_norm = msg_norm
        self.encode_edge = encode_edge
        self.bond_encoder = bond_encoder

        if msg_norm:
            self.msg_norm = MsgNorm(learn_msg_scale=learn_msg_scale)
        else:
            self.msg_norm = None

        if self.encode_edge:
            if self.bond_encoder:
                self.edge_encoder = BondEncoder(emb_dim=emb_dim)
            else:
                self.edge_encoder = torch.nn.Linear(edge_feat_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr=None, x_orig=None):
        '''
        x: input node embedding at this layer
        edge_index: usual
        edge_attr: usual
        x_orig: an older value of x, can be the input to the whole GCN or 
                edge_dist_basis in case of linegraph
        '''
        if self.encode_edge and edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = edge_attr

        m = self.propagate(edge_index, x=x, edge_attr=edge_emb, x_orig=x_orig)

        if self.msg_norm is not None:
            m = self.msg_norm(x, m)

        h = x + m
        out = self.mlp(h)

        return out

    def message(self, x_j, edge_attr=None, x_orig_i=None):
        if edge_attr is not None:
            # multiply or add?
            if self.emb_product:
                if self.emb_use_both:
                    # multiply with the original X
                    msg1 = x_j * self.linear_target_x(x_orig_i)
                    # project it once
                    msg2 = self.linear_intermediate(msg1)
                    # multiply with the edge feature
                    msg = msg2 * self.linear_edgeattr(edge_attr)
                else:
                    msg = x_j * edge_attr
            else:
                msg = x_j + edge_attr
        else:
            msg = x_j

        return self.msg_encoder(msg) + self.eps

    def update(self, aggr_out):
        return aggr_out
