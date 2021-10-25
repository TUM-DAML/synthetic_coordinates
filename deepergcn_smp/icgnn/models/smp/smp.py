import torch
import torch.nn.functional as F
import torch.nn as nn
from icgnn.models.smp.smp_layers import SMPLayer
from icgnn.models.smp.layers import GraphExtractor, EdgeCounter, BatchNorm
from icgnn.models.smp.misc import create_batch_info, map_x_to_u

import torch_scatter
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool



class SMP(torch.nn.Module):
    def __init__(self, num_input_features: int, num_edge_features: int, 
                 num_classes: int, num_layers: int,
                 hidden: int, residual: bool, use_edge_features: bool, shared_extractor: bool,
                 hidden_final: int, use_batch_norm: bool, use_x: bool, map_x_to_u: bool,
                 num_towers: int, simplified: bool):
        """ num_input_features: number of node features
            num_edge_features: number of edge features
            num_classes: output dimension
            hidden: number of channels of the local contexts
            residual: use residual connexion after each SMP layer
            use_edge_features: if False, edge features are simply ignored
            shared extractor: share extractor among layers to reduce the number of parameters
            hidden_final: number of channels after extraction of graph features
            use_x: for ablation study, run a MPNN instead of SMP
            map_x_to_u: map the initial node features to the local context. If false, node features are ignored
            num_towers: inside each SMP layers, use towers to reduce the number of parameters
            simplified: if True, the feature extractor has less layers.
        """
        super().__init__()
        self.map_x_to_u, self.use_x = map_x_to_u, use_x
        self.use_batch_norm = use_batch_norm
        self.edge_counter = EdgeCounter()
        self.num_classes = num_classes
        self.residual = residual
        self.shared_extractor = shared_extractor

        self.no_prop = GraphExtractor(in_features=num_input_features, out_features=hidden_final, use_x=use_x)
        self.initial_lin = nn.Linear(num_input_features, hidden)

        self.convs = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        for i in range(0, num_layers):
            self.convs.append(SMPLayer(in_features=hidden, num_towers=num_towers, out_features=hidden,
                                           edge_features=num_edge_features, use_x=use_x,
                                           use_edge_features=use_edge_features))
            self.batch_norm_list.append(BatchNorm(hidden, use_x) if i > 0 else None)

        # Feature extractors
        if shared_extractor:
            self.feature_extractor = GraphExtractor(in_features=hidden, out_features=hidden_final, use_x=use_x,
                                                    simplified=simplified)
        else:
            self.feature_extractors = torch.nn.ModuleList([])
            for i in range(0, num_layers):
                self.feature_extractors.append(GraphExtractor(in_features=hidden, out_features=hidden_final,
                                                              use_x=use_x, simplified=simplified))

        # Last layers
        self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, data, make_pred=True):
        """ data.x: (num_nodes, num_node_features)"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Compute information about the batch
        batch_info = create_batch_info(data, self.edge_counter)
        # Create the context matrix
        if self.use_x:
            assert x is not None
            u = x
        elif self.map_x_to_u:
            u = map_x_to_u(data, batch_info)
        else:
            u = data.x.new_zeros((data.num_nodes, batch_info['n_colors']))
            u.scatter_(1, data.coloring, 1)
            u = u[..., None]

        # Forward pass
        out = self.no_prop(u, batch_info)
        u = self.initial_lin(u)
        for i in range(len(self.convs)):
            conv = self.convs[i]
            bn = self.batch_norm_list[i]
            extractor = self.feature_extractor if self.shared_extractor else self.feature_extractors[i]
            if self.use_batch_norm and i > 0:
                u = bn(u)
            u = conv(u, edge_index, edge_attr, batch_info) + (u if self.residual else 0)
            global_features = extractor.forward(u, batch_info)
            out += global_features / len(self.convs)

        out = torch.relu(self.after_conv(out)) + out

        # make prediction, or return the activations?
        if make_pred:
            out = self.final_lin(out)
            assert out.shape[1] == 1
        return out 

    def __repr__(self):
        return self.__class__.__name__

class SMP_LineGraph(torch.nn.Module):
    def __init__(self, num_input_features: int, num_edge_features: int, 
                 num_classes: int, num_layers: int,
                 hidden: int, residual: bool, use_edge_features: bool, shared_extractor: bool,
                 hidden_final: int, use_batch_norm: bool, use_x: bool, map_x_to_u: bool,
                 num_towers: int, simplified: bool,
                 # extra args for linegraph message passing
                 lg_node_basis=4, lg_edge_basis=4,
                 graph_pooling='mean',):
        '''
        SMP on the linegraph
        '''
        super().__init__()
        self.smp = SMP(hidden_final + 1, lg_edge_basis, 
                 num_classes, num_layers,
                 hidden, residual, use_edge_features, shared_extractor,
                 hidden_final, use_batch_norm, use_x, map_x_to_u,
                 num_towers, simplified)

        self.node_features_encoder = torch.nn.Linear(num_input_features, hidden)

        # layer to embed the LG node message
        # 2 node embeddings + edge feature + dist basis dimension
        self.msg_emb_layer = nn.Linear(2 * hidden \
                                        + num_edge_features \
                                        + lg_node_basis, 
                                        hidden_final)

        self.pool = {
            'sum': global_add_pool,
            'mean': global_mean_pool,
            'max': global_max_pool
        }[graph_pooling]                                        

        # final prediction layer
        self.graph_pred_linear = torch.nn.Linear(hidden_final, num_classes)

    def forward(self, batch):
        # initial embedding of graph node feature
        h = self.node_features_encoder(batch.x_g.to(torch.float32))

        # construct the first message = h_uv || h_vu || edgeattr_uv
        msg1 = torch.index_select(h, dim=0, index=batch.edge_index_g[0])
        msg2 = torch.index_select(h, dim=0, index=batch.edge_index_g[1])
        msg_concat = torch.cat((msg1, msg2, batch.edge_attr_g.to(torch.float32), 
                                batch.x_lg), dim=-1)
        h = self.msg_emb_layer(msg_concat)

        # batch list of the linegraph - which linegraph does each *edge* belong to?
        # = the graph which the starting node belongs to
        start_nodes = batch.edge_index_g[0]

        batch_ndx_lg = batch.batch[start_nodes]

        # create the batch with the linegraph
        batch_lg = Batch(batch=batch_ndx_lg, x=h, edge_index=batch.edge_index_lg, 
                        edge_attr=batch.edge_attr_lg)

        # pass through SMP, get 1 pred per linegraph = 1 pred per graph
        # no need to aggregate or pool
        h_smp = self.smp(batch_lg, make_pred=False)
            
        return self.graph_pred_linear(h_smp)
