import torch
import torch.nn as nn
from icgnn.models.emb_basis_utils import get_global_emb_layer, get_local_emb_layer
from icgnn.models.smp.smp_layers import SMPLayer
from icgnn.models.smp.layers import GraphExtractor, EdgeCounter, BatchNorm
from icgnn.models.smp.misc import create_batch_info, map_x_to_u

from torch_geometric.data import Batch


class SMP(torch.nn.Module):
    def __init__(
        self,
        num_input_features: int,
        num_edge_features: int,
        num_classes: int,
        num_layers: int,
        hidden_final: int,
        emb_basis_global=True,
        emb_basis_local=True,
        emb_bottleneck=4,
    ):
        """num_input_features: number of node features
        num_edge_features: number of edge features
        num_classes: output dimension
        hidden_final: number of channels after extraction of graph features
        """
        super().__init__()
        self.edge_counter = EdgeCounter()

        self.num_classes = num_classes

        self.no_prop = GraphExtractor(
            in_features=num_input_features, out_features=hidden_final
        )
        hidden = 32
        self.initial_lin = nn.Linear(num_input_features, hidden)

        # create one global emb layer for the edgeattr
        self.global_emb_edge_attr = get_global_emb_layer(
            emb_basis_global, emb_basis_local, emb_bottleneck, num_edge_features, hidden
        )
        # create one local emb layer for each conv layer
        self.emb_edge_attrs = nn.ModuleList(
            [
                get_local_emb_layer(
                    emb_basis_global,
                    emb_basis_local,
                    emb_bottleneck,
                    num_edge_features,
                    hidden,
                )
                for _ in range(num_layers)
            ]
        )

        self.convs = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        for i in range(0, num_layers):
            self.convs.append(
                SMPLayer(
                    in_features=hidden,
                    num_towers=8,
                    out_features=hidden,
                    edge_features=hidden,
                )
            )
            self.batch_norm_list.append(BatchNorm(hidden) if i > 0 else None)

        # Feature extractors
        self.feature_extractor = GraphExtractor(
            in_features=hidden,
            out_features=hidden_final,
        )

        # Last layers
        self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, data):
        """data.x: (num_nodes, num_node_features)"""
        _, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Compute information about the batch
        batch_info = create_batch_info(data, self.edge_counter)
        # Create the context matrix
        u = map_x_to_u(data, batch_info)

        # global emb on the edge attr once
        edge_emb = self.global_emb_edge_attr(edge_attr)

        # Forward pass
        out = self.no_prop(u, batch_info)
        u = self.initial_lin(u)
        for i in range(len(self.convs)):
            conv = self.convs[i]
            bn = self.batch_norm_list[i]
            extractor = self.feature_extractor
            if i > 0:
                u = bn(u)

            # get the edge attr for this layer
            edge_emb_local = self.emb_edge_attrs[i](edge_emb)

            u = conv(u, edge_index, edge_emb_local, batch_info) + u

            global_features = extractor.forward(u, batch_info)
            out += global_features / len(self.convs)

        out = torch.relu(self.after_conv(out)) + out

        out = self.final_lin(out)
        return out

    def __repr__(self):
        return self.__class__.__name__


class SMP_LineGraph(torch.nn.Module):
    def __init__(
        self,
        num_input_features: int,
        num_edge_features: int,
        num_classes: int,
        num_layers: int,
        hidden_final: int,
        # extra args for linegraph message passing
        lg_node_basis=4,
        lg_edge_basis=4,
        emb_basis_global=True,
        emb_basis_local=True,
        emb_bottleneck=4,
    ):
        """
        SMP on the linegraph
        """
        super().__init__()
        self.edge_counter = EdgeCounter()

        self.num_classes = num_classes

        hidden = 32
        # graph x encoder
        self.node_features_encoder = torch.nn.Linear(num_input_features, hidden)
        # layer to embed the LG node message
        # 2 node embeddings + edge feature + dist basis dimension
        self.msg_emb_layer = nn.Linear(
            2 * hidden + num_edge_features + lg_node_basis, hidden_final
        )
        # on the linegraph
        lg_input_features = hidden_final + 1
        self.initial_lin = nn.Linear(lg_input_features, hidden)
        self.no_prop = GraphExtractor(
            in_features=lg_input_features, out_features=hidden_final
        )

        # create one global emb layer for the dist basis
        self.global_emb_dist_basis = get_global_emb_layer(
            emb_basis_global, emb_basis_local, emb_bottleneck, lg_node_basis, hidden
        )
        # create one local emb layer for each conv layer
        self.local_emb_dist_basis = nn.ModuleList(
            [
                get_local_emb_layer(
                    emb_basis_global,
                    emb_basis_local,
                    emb_bottleneck,
                    lg_node_basis,
                    hidden,
                )
                for _ in range(num_layers)
            ]
        )

        # create one global for angle basis
        self.global_emb_angle_basis = get_global_emb_layer(
            emb_basis_global, emb_basis_local, emb_bottleneck, lg_edge_basis, hidden
        )
        # create one local emb layer for each conv layer
        self.local_emb_angle_basis = nn.ModuleList(
            [
                get_local_emb_layer(
                    emb_basis_global,
                    emb_basis_local,
                    emb_bottleneck,
                    lg_edge_basis,
                    hidden,
                )
                for _ in range(num_layers)
            ]
        )

        self.convs = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        for i in range(0, num_layers):
            self.convs.append(
                SMPLayer(
                    in_features=hidden,
                    num_towers=8,
                    out_features=hidden,
                    edge_features=hidden,
                )
            )
            self.batch_norm_list.append(BatchNorm(hidden) if i > 0 else None)

        # Feature extractors
        self.feature_extractor = GraphExtractor(
            in_features=hidden,
            out_features=hidden_final,
        )

        self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

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

        # batch list of the linegraph - which linegraph does each *edge* belong to?
        # = the graph which the starting node belongs to
        start_nodes = batch.edge_index_g[0]
        batch_ndx_lg = batch.batch[start_nodes]

        # create the batch with the linegraph
        # pass through SMP, get 1 pred per linegraph = 1 pred per graph
        data = Batch(
            batch=batch_ndx_lg,
            x=h,
            edge_index=batch.edge_index_lg,
            edge_attr=batch.edge_attr_lg,
        )

        ### similar to SMP from here
        _, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Compute information about the batch
        batch_info = create_batch_info(data, self.edge_counter)
        # Create the context matrix
        u = map_x_to_u(data, batch_info)

        # global emb on the edge attr once
        edge_emb_global = self.global_emb_angle_basis(edge_attr)

        # Forward pass
        out = self.no_prop(u, batch_info)
        u = self.initial_lin(u)

        # globally embed the distance basis
        dist_emb_global = self.global_emb_dist_basis(batch.x_lg)

        for i in range(len(self.convs)):
            conv = self.convs[i]
            bn = self.batch_norm_list[i]
            extractor = self.feature_extractor
            if i > 0:
                u = bn(u)

            # local emb of the dist basis
            dist_emb_local = self.local_emb_dist_basis[i](dist_emb_global)

            # get the edge attr for this layer (angle basis)
            edge_emb_local = self.local_emb_angle_basis[i](edge_emb_global)

            # conv as usual
            u = (
                conv(
                    u, edge_index, edge_emb_local, batch_info, dist_basis=dist_emb_local
                )
                + u
            )
            global_features = extractor.forward(u, batch_info)
            out += global_features / len(self.convs)

        out = torch.relu(self.after_conv(out)) + out

        out = self.final_lin(out)
        return out
