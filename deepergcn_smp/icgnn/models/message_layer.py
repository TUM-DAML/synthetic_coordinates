import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class MessageLayer(MessagePassing):
    def __init__(
        self,
        node_channels,
        edge_channels,
        act_fn,
        dropout,
        bias=False,
        degree_norm=False,
        aggr="add",
        **kwargs
    ):
        """
        node_channels: the dimension of the node embedding
        edge_channels: the dimension of the edge embedding.
            eg: cosine basis of distance with 4 components
        act_fn: activation function
        dropout: dropout probability
        bias: use bias?
        degree_norm: normalize the message by the degrees of neighboring nodes
        aggr: type of aggregation to perform
        """
        super().__init__(aggr=aggr, **kwargs)

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.act_fn = act_fn
        self.dropout = nn.Dropout(dropout)

        # apply degree normalization during message passing (GCNConv equation)
        self.degree_norm = degree_norm

        self.fc_edge = nn.Linear(edge_channels, node_channels, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(node_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph_x, graph_edge_index, graph_edge_attr=None):
        """
        Initial call to start propagating messages (calls propagate)

        graph_x: embedding of the graph vertex
        graph_edge_index: edges of the graph
        graph_edge_attr: graph.edge_attr
        """
        edge_emb = self.act_fn(self.fc_edge(graph_edge_attr))

        if self.degree_norm:
            # normalize the embeddings by the degree of the incoming, outgoing vertices
            row, col = graph_edge_index
            deg = degree(row, graph_x.size(0), dtype=graph_x.dtype) + 1
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # broadcast the norm into the embeddings
            edge_emb = norm[:, None] * edge_emb

        return self.propagate(graph_edge_index, x=graph_x, edge_emb=edge_emb)

    def message(self, x_j, edge_emb):
        """
        Constructs messages to node i for each edge (j, i) if flow is "source_to_target"
        else the other way

        x_j: embedding of a vertex
        edge_emb: embedding of a linegraph edge (2 adjacent edges in original graph)
        """
        return edge_emb * x_j

    def update(self, aggr_out):
        """
        Updates node embeddings for each node i

        aggr_out: output of aggregation
        """
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.node_channels, self.edge_channels
        )
