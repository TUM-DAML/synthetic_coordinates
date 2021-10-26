"""
ICGNN version of the TUDataset
Includes the original graphs in the TUDataset
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/tu_dataset.html#TUDataset

along with VERSE embeddings, linegraph, distances between nodes
and angles between edges
"""

import os, os.path as osp
import copy
from tqdm import tqdm
from functools import partial

print = partial(print, flush=True)

from multiprocessing import Pool

import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
import torch_geometric.transforms as T

from ..models.embeddings import create_embedding
from ..models.basis import get_dist_basis, get_cos_basis


def graph_to_icgnn(graph, emb_dim, num_dist_basis=4, num_cos_basis=4):
    embedding = torch.Tensor(create_embedding(graph, emb_dim, "verse"))

    graph.emb = embedding

    emb_diff = torch.index_select(embedding, dim=0, index=graph.edge_index[0])
    emb_diff -= torch.index_select(embedding, dim=0, index=graph.edge_index[1])
    emb_dist = torch.norm(emb_diff, p=2, dim=1)

    # TODO: retain the original edge_attr, dont overwrite it!
    graph.edge_attr = get_dist_basis(emb_dist, num_dist_basis)

    graph_copy = Data(
        edge_index=copy.deepcopy(graph.edge_index), num_nodes=graph.num_nodes
    )
    transform = T.Compose([T.LineGraph(force_directed=True), T.Constant()])
    linegraph = transform(graph_copy)

    noloop_mask = ~(
        graph.edge_index[0, linegraph.edge_index[0]]
        == graph.edge_index[1, linegraph.edge_index[1]]
    )
    line_idx_noloop = graph.edge_index.new_empty((2, noloop_mask.sum()))
    line_idx_noloop[0] = torch.masked_select(linegraph.edge_index[0], noloop_mask)
    line_idx_noloop[1] = torch.masked_select(linegraph.edge_index[1], noloop_mask)
    linegraph.edge_index = line_idx_noloop

    emb_diff_ji = torch.index_select(
        -emb_diff, dim=0, index=linegraph.edge_index[0]
    )  # Ri <- Rj
    emb_diff_jk = torch.index_select(
        emb_diff, dim=0, index=linegraph.edge_index[1]
    )  # Rj -> Rk
    emb_dist_ji = torch.norm(emb_diff_ji, p=2, dim=1)
    emb_dist_jk = torch.norm(emb_diff_jk, p=2, dim=1)
    emb_angle = torch.acos(
        (emb_diff_ji * emb_diff_jk).sum(-1) / (emb_dist_ji * emb_dist_jk)
    )

    linegraph.edge_attr = get_cos_basis(emb_angle, num_cos_basis)

    return graph, linegraph


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


def data_to_storage(data):
    keys = data.keys
    data_new = data.__class__()

    for key in data.keys:
        item = data[key]
        if torch.is_tensor(item):
            data_new[key] = item.clone().detach()
        elif isinstance(item, int) or isinstance(item, float):
            data_new[key] = torch.tensor(item)
    return data_new


def data_to_icgnndata(graph, emb_dim, write_file=False, base_path=None):
    if isinstance(graph, tuple):
        ndx, graph = graph

    try:
        graph, linegraph = graph_to_icgnn(graph, emb_dim=emb_dim)
    except:
        print(f"Could not create VERSE for graph {ndx}")
        return

    icgnn_data = ICGNN_Data(
        x_g=graph.x,
        edge_index_g=graph.edge_index,
        x_emb=graph.emb,
        edge_attr_g=graph.edge_attr,
        x_lg=linegraph.x,
        edge_index_lg=linegraph.edge_index,
        edge_attr_lg=linegraph.edge_attr,
        y=graph.y,
    )

    if write_file:
        out_path = osp.join(base_path, f"data_{ndx}.pt")
        torch.save(data_to_storage(icgnn_data), out_path)
    else:
        return icgnn_data


class ICGNNDataset(InMemoryDataset):
    def __init__(
        self, base_dataset=None, emb_dim=64, use_multiproc=False, transform=None
    ):
        """
        base_dataset: the TUDataset object from which this new dataset is created
        emb_dim: embedding dimension for VERSE
        use_multiproc: use multi processing to create the dataset
        transform: the transform to apply on each sample
        """
        self.use_multiproc = use_multiproc
        self.emb_dim = emb_dim
        self.base_dataset = base_dataset
        self.root = self.base_dataset.root
        # creates the processed .pt file if it doesn't exist
        super().__init__(base_dataset.root, transform=transform)
        # we dont need this anymore after processing, clear it
        self.base_dataset = None
        # loads the .pt file
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        """
        Save the processed dataset to the same directory
        """
        return osp.join(self.root, "processed")

    @property
    def raw_file_names(self):
        # no file inputs. directly take a Dataset object
        return []

    @property
    def processed_file_names(self):
        """
        Output a single file
        """
        return ["icgnn_data.pt"]

    def get(self, idx):
        from itertools import repeat

        """
        copied from original source code
        and edited to "detach" the tensors in the graph
        so that we can use multiproccessing
        """
        # create an empty object
        data = self.data.__class__()

        # copy the num nodes attribute
        if hasattr(self.data, "__num_nodes__"):
            data.num_nodes = self.data.__num_nodes__[idx]

        # copy everything
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()

            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)

            # NEW:
            data[key] = item[s].detach()
        return data

    def process(self):
        # create a new ICGNN_Data list from the TU Dataset
        data_list = []
        print(f"Processing {len(self.base_dataset)} graphs")

        # multiprocessing
        if self.use_multiproc:
            # fix the emb_dim for all function calls
            partial_func = partial(data_to_icgnndata, emb_dim=self.emb_dim)

            chunksize = 8
            n_proc = 8
            print(f"Using {n_proc} processes, chunksize: {chunksize}")
            with Pool(processes=n_proc) as pool:
                for result in tqdm(
                    pool.imap(
                        func=partial_func,
                        iterable=self.base_dataset,
                        chunksize=chunksize,
                    ),
                    total=len(self.base_dataset),
                ):
                    data_list.append(data_to_storage(result))
        else:
            print(f"Using a single process")
            # one at a time
            for (ndx, graph) in enumerate(tqdm(self.base_dataset)):
                result = data_to_icgnndata((ndx, graph), emb_dim=self.emb_dim)
                data_list.append(result)

        # collate to get data and slices
        data, slices = self.collate(data_list)
        # save to file
        torch.save((data, slices), self.processed_paths[0])


class LargeICGNNDataset(Dataset):
    def __init__(
        self,
        base_dataset=None,
        emb_dim=32,
        use_multiproc=True,
        transform=None,
        ignore_ndx=[],
    ):
        """
        base_dataset: the TUDataset object from which this new dataset is created
        """
        self.use_multiproc = use_multiproc
        self.emb_dim = emb_dim
        self.base_dataset = base_dataset
        self.root = self.base_dataset.root

        # indices of graphs in the original dataset
        # for which there is no ICGNN processed data
        # TODO: why cant we create VERSE for these?
        self.ignore_ndx = ignore_ndx

        # store length as an integer
        # dont need to refer to the dataset next time
        self._length = len(self.base_dataset)

        # creates the processed .pt files if they doesn't exist
        super().__init__(base_dataset.root, transform=transform)

        # dont need this any more after processing
        # TODO: dont need this the second time the dataset object is created
        # only the first time
        self.base_dataset = None

    @property
    def processed_dir(self):
        """
        Save the processed dataset to the same directory
        """
        return osp.join(self.root, "processed", "icgnn_graphs")

    @property
    def raw_file_names(self):
        # no file inputs. directly take a Dataset object
        return []

    @property
    def processed_file_names(self):
        """
        Output multiple files - 1 for each graph
        """
        filenames = [
            f"data_{i}.pt"
            for i in range(len(self.base_dataset))
            if i not in self.ignore_ndx
        ]
        return filenames

    def len(self):
        return self._length

    def get(self, idx):
        if idx in self.ignore_ndx:
            idx = 0
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data

    def process(self):
        # create a new ICGNN_Data list from the TU Dataset
        print("Creating a large dataset!")

        print(f"Processing {len(self.base_dataset)} graphs")

        # multiprocessing
        if self.use_multiproc:
            # fix the emb_dim for all function calls
            partial_func = partial(
                data_to_icgnndata,
                emb_dim=self.emb_dim,
                write_file=True,
                base_path=self.processed_dir,
            )

            chunksize = 8
            n_proc = 8
            print(f"Using {n_proc} processes, chunksize: {chunksize}")
            with Pool(processes=n_proc) as pool:
                for _ in tqdm(
                    pool.imap(
                        func=partial_func,
                        iterable=enumerate(self.base_dataset),
                        chunksize=chunksize,
                    ),
                    total=len(self.base_dataset),
                ):
                    pass
        else:
            print(f"Using a single process")
            # one at a time
            for i, graph in enumerate(tqdm(self.base_dataset)):
                result = data_to_icgnndata(graph, emb_dim=self.emb_dim)
                torch.save(
                    data_to_storage(result),
                    osp.join(self.processed_dir, f"data_{i}.pt"),
                )
