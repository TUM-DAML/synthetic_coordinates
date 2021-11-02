__author__ = "Stefan Weißenberger and Johannes Klicpera"

import os, os.path as osp

import numpy as np
from torch_geometric.transforms.compose import Compose
from tqdm import tqdm

import torch
from torch.utils.data import Subset
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import ZINC
import torch_geometric.transforms as T

# stanford ogb
from ogb.graphproppred import PygGraphPropPredDataset

from icgnn.data_utils.io_utils import read_jsonl
from icgnn.data_utils.qm9 import qm9_gnnfilm_to_pyg, QM9_GNNFilm
from icgnn.transforms.zinc import OneHotNodeEdgeFeatures, ZINC_Reshape_Target

DATA_PATH = "../data"


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

    if name == "ogbg-molhiv":
        dataset = PygGraphPropPredDataset(
            root=DATA_PATH, name="ogbg-molhiv", transform=transform
        )
        split_idx = dataset.get_idx_split()

        if quick_run:
            print("----quick run----")
            n_select = 128
            split_idx["train"] = torch.LongTensor(range(n_select))
            split_idx["valid"] = torch.LongTensor(range(n_select))
            split_idx["test"] = torch.LongTensor(range(n_select))
        train_set = Subset(dataset, split_idx["train"].numpy().tolist())
        val_set = Subset(dataset, split_idx["valid"].numpy().tolist())
        test_set = Subset(dataset, split_idx["test"].numpy().tolist())
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

    else:
        raise NotImplementedError

    return train_set, val_set, test_set
