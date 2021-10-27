"""
ICGNN version of the TUDataset
Includes the original graphs in the TUDataset
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/tu_dataset.html#TUDataset

along with VERSE embeddings, linegraph, distances between nodes
and angles between edges
"""

from functools import partial

print = partial(print, flush=True)

from torch_geometric.data import Data


class ICGNN_Data(Data):
    def __init__(
        self,
        x_g=None,
        edge_index_g=None,
        edge_attr_g=None,
        x_emb=None,
        x_lg=None,
        edge_index_lg=None,
        edge_attr_lg=None,
        y=None,
    ):
        """
        *_g: properties of the graph
        *_lg: properties of the linegraph

        See tutorial here:
        https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html#pairs-of-graphs
        """
        super().__init__()

        self.edge_index_g = edge_index_g
        self.x_g = x_g
        self.x_emb = x_emb
        self.edge_attr_g = edge_attr_g

        self.edge_index_lg = edge_index_lg
        self.x_lg = x_lg
        self.edge_attr_lg = edge_attr_lg

        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_g":
            return self.x_g.size(0)
        if key == "edge_index_lg":
            return self.x_lg.size(0)
        else:
            return super().__inc__(key, value)
