from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import get_laplacian

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdmolops import FastFindRings

from ogb.utils.features import get_atom_feature_dims

from icgnn.transforms.rdkit_utils import (
    get_chiral_tag,
    get_formal_charge,
    get_hybridization,
    get_aromatic,
    bond_ndx_to_bond,
    get_bond_stereo,
    get_conjugated,
)


def extract_node_feature(data, reduce="add"):
    if reduce in ["mean", "max", "add"]:
        data.x = scatter(
            data.edge_attr,
            data.edge_index[0],
            dim=0,
            dim_size=data.num_nodes,
            reduce=reduce,
        )
    else:
        raise Exception("Unknown Aggregation Type")
    return data


class Graph_To_Mol:
    """
    Convert OGB graph (hiv or pcba) to an RDKit molecule
    """

    def __init__(self):
        pass

    def __call__(self, graph):
        # create a read/write molecule
        mol = Chem.RWMol()

        # set atom properties and add to the molecule
        for atom_ndx, feature in enumerate(graph.x):
            feature = feature.numpy().tolist()
            atom = Chem.Atom(feature[0] + 1)
            atom.SetChiralTag(get_chiral_tag(feature[1]))
            atom.SetFormalCharge(get_formal_charge(feature[3]))
            atom.SetNumRadicalElectrons(feature[5])
            atom.SetHybridization(get_hybridization(feature[6]))
            atom.SetIsAromatic(get_aromatic(feature[7]))
            mol.AddAtom(atom)

        bond_types = graph.edge_attr[:, 0].numpy()
        bonds = list(map(bond_ndx_to_bond, bond_types))

        # create a bond, set its properties and add to the molecule
        for ndx, (i, j) in enumerate(graph.edge_index.T):
            feature = graph.edge_attr[ndx].numpy().tolist()
            bond = mol.GetBondBetweenAtoms(i.item(), j.item())
            if bond is None:
                mol.AddBond(i.item(), j.item(), bonds[ndx])
                bond = mol.GetBondBetweenAtoms(i.item(), j.item())
                bond.SetStereo(get_bond_stereo(feature[1]))
                bond.SetIsConjugated(get_conjugated(feature[2]))

        # cleanup the molecule
        mol.UpdatePropertyCache()
        FastFindRings(mol)

        return (graph, mol)


class OneHot_Mol:
    """
    One hot encoding for molhiv and molpcba graphs
    """

    def __init__(self):
        # inbuilt function to get the max number of classes for each feature
        self.num_classes = get_atom_feature_dims()
        # 0: atomic number - int
        # 1: chirality - onehot
        # 2: degree - int
        # 3: charge - int
        # 4: numH - int
        # 5: num radical - int
        # 6: hybridization - one hot
        # 7: aromatic - binary one hot = int (0/1)
        # 8: is in ring - binary one hot = int (0/1)

        # only these features need one hot encoding
        self.one_hot_features = [0, 1, 6]

        # edge features which need to be one hot encoded
        # bond type (single, double..) and stereo (none, z, e, cis, trans, any)
        self.one_hot_edge_features = [0, 1]

        # feature which dont need one hot encoding
        self.regular_features = [
            n for n in range(len(self.num_classes)) if n not in self.one_hot_features
        ]

        # total input feature dimension
        total_dim = sum([self.num_classes[i] for i in self.one_hot_features]) + len(
            self.regular_features
        )
        print(f"graph.x one hot encoding with {total_dim} classes")

    def __call__(self, data):
        """
        data: data_utils.icgnn_dataset.ICGNN_Data

        convert the indices in data.x_g into a one hot encoding
        """
        # accumulate the one hot encoding of each feature here
        # then concat
        new_data = deepcopy(data)

        # keep non-onehot features as-is
        new_x = [new_data.x_g[:, self.regular_features]]

        new_x += [
            F.one_hot(
                new_data.x_g[:, feature_ndx], num_classes=self.num_classes[feature_ndx]
            )
            for feature_ndx in self.one_hot_features
        ]

        new_data.x_g = torch.cat(new_x, dim=-1)

        return new_data


class Laplacian_PositionEnc(object):
    """
    Positional encoding based on graph Laplacian
    see paper: Benchmarking Graph Neural Networks
    https://arxiv.org/pdf/2003.00982.pdf
    """

    def __init__(self, x_dim=135, k=16):
        """
        x_dim: dimension of the graph node feature
        k: pick bottom k eigenvectors
        """
        self.k = k
        self.linear = nn.Linear(k, x_dim)

    def __call__(self, data):
        """
        data: data_utils.icgnn_dataset.ICGNN_Data
        add positional encoding to the graph
        """
        print(data)
        # get full adj matrix
        num_nodes = data.x_g.shape[0]
        adj_size = torch.Size([num_nodes, num_nodes])

        lap_edges, lap_val = get_laplacian(data.edge_index_g, normalization="sym")

        L = torch.sparse_coo_tensor(lap_edges, lap_val, adj_size).to_dense()
        result = torch.eig(L, eigenvectors=True)
        eigval, eigvec = result.eigenvalues[:, 0], result.eigenvectors

        # select the first k dimensions of each eigenvector
        eigvec = eigvec[: self.k, :]

        # sort eigenvalues
        sorted_vals, sorted_ndx = torch.sort(eigval)

        # pick the lowest k eigvals and corresponding vectors
        bottom_k_ndx = sorted_ndx[: self.k]
        bottom_k_eigvec = eigvec[:, bottom_k_ndx]
        bottom_k_eigval = sorted_vals[: self.k]
        # normalize by the eigenvalue
        norm_k_eigvec = bottom_k_eigvec / bottom_k_eigval

        # randomly flip sign - get random -1, +1 vector
        rands = torch.rand_like(bottom_k_eigval)
        rand_sign = ((rands > 0.5) * 1 - 0.5) * 2
        # multiply with eigvals
        rand_sign_eigvals = bottom_k_eigval * rand_sign

        # choose from these and add to graph.x
        choice = torch.randint(self.k, (num_nodes,))
        eigvecs_fulldim = rand_sign_eigvals.T[choice]
        pos_enc = self.linear(eigvecs_fulldim)

        # add positional encoding to feature
        data.x_g += pos_enc

        return data
