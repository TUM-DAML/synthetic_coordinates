__author__ = "Stefan Weißenberger and Johannes Klicpera"
__license__ = "MIT"


import os, os.path as osp

import numpy as np
from functools import partial
from torch_geometric.transforms.compose import Compose
from tqdm import tqdm

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Coauthor,
    CoraFull,
    TUDataset,
    ZINC,
    QM9,
)
import torch_geometric.transforms as T

# stanford ogb
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset

from icgnn.data_utils.io_utils import read_jsonl
from icgnn.data_utils.qm9 import qm9_gnnfilm_to_pyg, QM9_GNNFilm
from .icgnn_dataset import ICGNNDataset, LargeICGNNDataset
from ..train_utils.seeds import development_seed
from icgnn.transforms.ogb import extract_node_feature
from icgnn.transforms.zinc import OneHotNodeEdgeFeatures, ZINC_Reshape_Target
from icgnn.train_utils.ogb_utils import get_vocab_mapping, encode_y_to_arr


DATA_PATH = "/nfs/students/yeshwant/datasets"
# DATA_PATH = '/nfs/staff-ssd/klicpera/datasets_icgnn'

TU_DATASETS = [
    "PROTEINS",
    "DD",
    "ENZYMES",
    "NCI1",
    "COLLAB",
    "IMDB-MULTI",
    "IMDB-BINARY",
    "REDDIT-BINARY",
    "REDDIT-MULTI-5K",
]

OGB_DATASETS = ["ogbg-molhiv", "ogbg-molpcba", "ogbg-ppa", "ogbg-code"]

LARGE_DATASETS = ["ogbg-molpcba", "ogbg-ppa", "ogbg-code" "COLLAB"]

# graphs to ignore in large datasets
IGNORE_NDX = {
    "ogbg-molhiv": [],
    "ogbg-molpcba": [
        55970,
        66946,
        81615,
        174380,
        244480,
        304420,
        338706,
        382939,
        389447,
        426549,
        430811,
    ],
    "ogbg-ppa": [],
    "ogbg-code": [],
}


def get_transformed_dataset(dataset):
    """
    Iterate over a torch Dataset so that transforms are applied
    Catch possible errors and discard samples
        eg: due to RDKit transforms
    so that the remaining samples are all good ones
    """
    data_list = []
    for ndx in tqdm(range(len(dataset)), miniters=200, maxinterval=np.inf):
        try:
            data_list.append(dataset[ndx])
        except (ValueError, RuntimeError, IndexError, KeyError):
            pass

    return data_list


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def get_graphcls_dataset(
    name: str,
    split=[0.6, 0.2, 0.2],
    icgnn=False,
    emb_dim=None,
    transform=None,
    quick_run=False,
) -> InMemoryDataset:
    """
    name: string indicating the name of the dataset used in the config.yml
    split: train, val, test split
    icgnn: bool, return the ICGNN version of the dataset which has VERSE embeddings
            and linegraph with distances, angles
    emb_dim: dimension of the VERSE embedding for ICGNN datasets
    transform: transform to apply on data samples
    """
    path = os.path.join(DATA_PATH, name)

    # with node labels
    if name in ("PROTEINS", "DD", "ENZYMES", "NCI1"):
        dataset = TUDataset(path, name=name, transform=transform)
    # no node labels, add local degree profile
    elif name in (
        "COLLAB",
        "IMDB-BINARY",
        "IMDB-MULTI",
        "REDDIT-BINARY",
        "REDDIT-MULTI-5K",
    ):
        t_composed = T.Compose(
            [
                T.LocalDegreeProfile(),
                transform,
            ]
        )
        dataset = TUDataset(path, name=name, transform=t_composed)
    elif name == "ogbg-molhiv":
        dataset = PygGraphPropPredDataset(
            root=DATA_PATH, name="ogbg-molhiv", transform=transform
        )
    elif name == "ogbg-molpcba":
        dataset = PygGraphPropPredDataset(
            root=DATA_PATH, name="ogbg-molpcba", transform=transform
        )
    elif name == "ogbg-ppa":
        print("(transform) PPA dataset: add edge features to get node feature")
        # add all neighboring edge features to get node feature
        extract_node_feature_func = partial(extract_node_feature, reduce="add")
        dataset = PygGraphPropPredDataset(
            root=DATA_PATH, name="ogbg-ppa", transform=extract_node_feature_func
        )
    elif name == "ogbg-code":
        # fixed vocab size and max seq len
        max_seq_len, num_vocab = 5, 5000
        # don't specify the transform now, set it later
        dataset = PygGraphPropPredDataset(root=DATA_PATH, name="ogbg-code")
        ### building vocabulary for sequence predition. Only use training data.
        split_idx = dataset.get_idx_split()
        vocab2idx, _ = get_vocab_mapping(
            [dataset.data.y[i] for i in split_idx["train"]], num_vocab
        )
        # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
        # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
        dataset.transform = T.Compose(
            [lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len), transform]
        )
    elif name == "ZINC":
        # node feature dim = 28, corresponds to use_x=False from original code
        # edge feature dim = bond type options = 3
        pre_transform = OneHotNodeEdgeFeatures(28, 3)

        t_composed = Compose([ZINC_Reshape_Target(), transform])

        train_set = ZINC(
            root=path,
            subset=True,
            split="train",
            pre_transform=pre_transform,
            transform=t_composed,
        )
        val_set = ZINC(
            root=path,
            subset=True,
            split="val",
            pre_transform=pre_transform,
            transform=t_composed,
        )
        test_set = ZINC(
            root=path,
            subset=True,
            split="test",
            pre_transform=pre_transform,
            transform=t_composed,
        )

        if quick_run:
            train_set = Subset(train_set, range(128))
            val_set = Subset(val_set, range(128))
            test_set = Subset(test_set, range(128))
    elif name == "QM9":
        # Use GNN FILM dataset, not pytorch geometric
        root = os.path.join(DATA_PATH, "qm9")

        train_raw = list(read_jsonl(osp.join(root, "train.jsonl.gz")))
        val_raw = list(read_jsonl(osp.join(root, "valid.jsonl.gz")))
        test_raw = list(read_jsonl(osp.join(root, "test.jsonl.gz")))

        if quick_run:
            train_raw = train_raw[:128]
            val_raw = val_raw[:128]
            test_raw = test_raw[:128]

        train_set = QM9_GNNFilm(
            list(map(qm9_gnnfilm_to_pyg, train_raw)), transform=transform
        )
        val_set = QM9_GNNFilm(
            list(map(qm9_gnnfilm_to_pyg, val_raw)), transform=transform
        )
        test_set = QM9_GNNFilm(
            list(map(qm9_gnnfilm_to_pyg, test_raw)), transform=transform
        )

    # elif name == 'QM9':
    #     dataset = QM9(root=path, transform=transform)

    #     train_set = Subset(dataset, range(100_000))
    #     val_set = Subset(dataset, range(100_000, 110_000))
    #     test_set = Subset(dataset, range(110_000, len(dataset)))

    #     if quick_run:
    #         train_set = Subset(train_set, range(128))
    #         val_set = Subset(val_set, range(128))
    #         test_set = Subset(test_set, range(128))
    else:
        raise NotImplementedError

    # use multiprocessing for large datasets
    # TODO: always use multiproc?
    # multiproc = name in ('ogbg-molpcba', 'ogbg-ppa')

    # OGB datasets - keep the split
    if name in OGB_DATASETS:
        split_idx = dataset.get_idx_split()
    # now convert to ICGNN dataset if specified
    if icgnn:
        if name in LARGE_DATASETS:
            ignore_ndx = [] if name not in IGNORE_NDX else IGNORE_NDX[name]
            # out of memory (disk) dataset - 1 file per graph
            dataset = LargeICGNNDataset(
                base_dataset=dataset,
                use_multiproc=True,
                emb_dim=emb_dim,
                transform=transform,
                ignore_ndx=ignore_ndx,
            )
        else:
            # in memory dataset
            dataset = ICGNNDataset(
                base_dataset=dataset,
                use_multiproc=False,
                emb_dim=emb_dim,
                transform=transform,
            )
    # TU Datasets: create our own split
    if name in TU_DATASETS:
        dataset = dataset.shuffle()
        train, val, test = split
        n = len(dataset)
        train_ndx = int(train * n)
        val_ndx = int((train + val) * n)

        train_set = dataset[:train_ndx]
        val_set = dataset[train_ndx:val_ndx]
        test_set = dataset[val_ndx:]
    # use existing split
    elif name in OGB_DATASETS:
        # remove the bad samples
        ignore_ndx = set(IGNORE_NDX[name])
        print(f"Ignoring {len(ignore_ndx)} bad samples in dataset")
        if quick_run:
            print("----quick run----")
            n_select = 32
            split_idx["train"] = torch.LongTensor(range(n_select))
            split_idx["valid"] = torch.LongTensor(range(n_select))
            split_idx["test"] = torch.LongTensor(range(n_select))
        train_set = Subset(
            dataset, list(set(split_idx["train"].numpy().tolist()) - ignore_ndx)
        )
        val_set = Subset(
            dataset, list(set(split_idx["valid"].numpy().tolist()) - ignore_ndx)
        )
        test_set = Subset(
            dataset, list(set(split_idx["test"].numpy().tolist()) - ignore_ndx)
        )

    return train_set, val_set, test_set


def get_dataset(name: str, use_lcc: bool = True, normalize=True) -> InMemoryDataset:
    path = os.path.join(DATA_PATH, name)
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures)
    elif name == "Cora-Full":
        dataset = CoraFull(path)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(path, name)
    elif name == "CoauthorCS":
        dataset = Coauthor(path, "CS")
    elif name == "CoauthorPhysics":
        dataset = Coauthor(path, "Physics")
    elif name == "ogbn-proteins":
        dataset = PygNodePropPredDataset(name=name)
    else:
        raise Exception("Unknown dataset.")

    if use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        )
        dataset.data = data

    return dataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [
            n for n in neighbors if n not in visited_nodes and n not in queued_nodes
        ]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def set_train_val_test_split(
    seed: int, data: Data, num_development: int = 1500, num_per_class: int = 20
) -> Data:
    # set the random state
    rnd_state = np.random.RandomState(development_seed)
    # the total number of vertices
    num_nodes = data.y.shape[0]
    # select the nodes to be used for train+test
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    # put the remaining in the test set
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    # now choose the train set
    train_idx = []
    rnd_state = np.random.RandomState(seed)

    # for each class
    for c in range(data.y.max() + 1):
        # first choose all the vertices in this class
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        # the number of nodes to choose for this class in the train set
        # some datasets have very fews nodes in a class
        num_to_choose = min(len(class_idx), num_per_class)
        train_idx.extend(rnd_state.choice(class_idx, num_to_choose, replace=False))
    # everything in development that's not in train is the val set
    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data
