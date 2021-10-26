"""
Functionality using RDKit on molecule graphs
eg: find all-pair distances between atoms
"""
import numpy as np

import torch

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Pharm3D import EmbedLib
from rdkit.Chem import rdDistGeom as molDG

from ..models.basis import get_dist_basis
from icgnn.models.embeddings import pairwise_distance
from icgnn.transforms.misc import get_radius_graph_from_distmat, set_or_append


class Set_Edge_Dist:
    """
    Set data.edge_dist and data.edge_dist_basis
    Can be used later as edge feature

    Assumes data.distances is already set to nxn matrix
    """

    def __init__(self, num_dist_basis=4, dist_basis_type="gaussian"):
        self.num_dist_basis = num_dist_basis
        self.dist_basis_type = dist_basis_type

    def __call__(self, data):
        if hasattr(data, "edge_dist_basis"):
            return data

        # (E, 2) edge index, each row is (i,j) = edge from i to j
        ndx = data.edge_index.T
        # get the distance for every edge
        edge_lengths = torch.Tensor(data.distances[ndx[:, 0], ndx[:, 1]])
        data.edge_dist = torch.Tensor(edge_lengths)
        # convert to a basis function representation
        data.edge_dist_basis = get_dist_basis(
            edge_lengths, self.dist_basis_type, self.num_dist_basis
        )

        return data


class Set_Distance_Matrix:
    """
    Use inbuilt 3d distance matrix function
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # sample contains the graph and molecule
        graph, mol = sample

        mol = Chem.AddHs(mol)
        # ETKDG is the default conformation generation method
        Chem.EmbedMolecule(mol, maxAttempts=1000)
        mol = Chem.RemoveHs(mol)

        graph.distances = Chem.Get3DDistanceMatrix(mol).astype(np.float32)

        return graph


class Set_3DCoord_Distance:
    """
    Calculate 3d coordinates, then pairwise distance
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # sample contains the graph and molecule
        graph, mol = sample

        mol = Chem.AddHs(mol)
        Chem.EmbedMolecule(mol, maxAttempts=1000)
        Chem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)

        coords = torch.Tensor(mol.GetConformers()[0].GetPositions())
        graph.distances = pairwise_distance(coords, coords)

        return graph


class Set_Pharm3D_Distance:
    """
    3D coordinates with Pharm3D
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # sample contains the graph and molecule
        graph, mol = sample

        mol = Chem.AddHs(mol)
        bm = molDG.GetMoleculeBoundsMatrix(mol)
        EmbedLib.EmbedMol(mol, bm)
        EmbedLib.OptimizeMol(mol, bm)
        mol = Chem.RemoveHs(mol)

        coords = torch.Tensor(mol.GetConformers()[0].GetPositions())
        graph.distances = pairwise_distance(coords, coords)

        return graph


class Set_2DCoord_Distance:
    """
    Get 2D coordinates, then distance
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # sample contains the graph and molecule
        graph, mol = sample
        mol = Chem.AddHs(mol)
        Chem.Compute2DCoords(mol)
        mol = Chem.RemoveHs(mol)

        coords = torch.Tensor(mol.GetConformers()[0].GetPositions())
        graph.distances = pairwise_distance(coords, coords)

        return graph


def get_upper_tri_distances(bmat):
    """
    Use the upper triangular part of the bounds matrix
    to get an all-pairs distances matrix
    by reflecting the upper tri part about the diagonal

    bmat: bounds matrix from RDKit
    """
    # use only the upper bounds
    # reflect it across the diagonal
    upper = np.triu(bmat)
    lower = upper.T
    # don't add the diagonal twice, set it to 0
    np.fill_diagonal(lower, 0)

    distances = lower + upper

    return distances


class Set_BoundsMatUpper_Distance:
    """
    Get bounds matrix with lower, upper bound
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # sample contains the graph and molecule
        graph, mol = sample
        mol = Chem.AddHs(mol)

        bm = molDG.GetMoleculeBoundsMatrix(mol)
        graph.distances = torch.Tensor(get_upper_tri_distances(bm))

        return graph


class Set_BoundsMatLower_Distance:
    """
    Get bounds matrix with lower, upper bound
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # sample contains the graph and molecule
        graph, mol = sample
        mol = Chem.AddHs(mol)

        bm = molDG.GetMoleculeBoundsMatrix(mol)
        graph.distances = torch.Tensor(get_upper_tri_distances(bm.T))

        return graph


class Set_BoundsMatBoth_Distance:
    """
    Get bounds matrix with lower, upper bound
    """

    def __init__(
        self,
        num_dist_basis=4,
        dist_basis_type="gaussian",
        metric_graph_cutoff=None,
        metric_graph_edgeattr=None,
    ):
        """
        metric_graph_cutoff: set new edges within this distance
        metric_graph_edgeattr:
            const: set constant edge attr to the new edges
            keep: keep existing edge attrs for old edges, const value for new edges
        """
        self.num_dist_basis = num_dist_basis
        self.dist_basis_type = dist_basis_type

        self.metric_graph_cutoff = metric_graph_cutoff
        self.metric_graph_edgeattr = metric_graph_edgeattr

        if self.metric_graph_cutoff:
            print(
                f"(transform) BM-both: Using metric graph. Edge attr method: {self.metric_graph_edgeattr}"
            )

    def __call__(self, sample):
        # sample contains the graph and molecule
        graph, mol = sample

        bm = molDG.GetMoleculeBoundsMatrix(mol)

        max_dist = get_upper_tri_distances(bm)
        min_dist = get_upper_tri_distances(bm.T)

        if self.metric_graph_cutoff:
            # dim of the existing edge attr
            old_edge_ndx = graph.edge_index.clone()
            graph.edge_index = torch.LongTensor(
                get_radius_graph_from_distmat(min_dist, self.metric_graph_cutoff)
            )
            edge_attr_dim = graph.edge_attr.shape[-1]
            num_new_edges = graph.edge_index.shape[-1]

            # replace all edge attr with zeros
            if self.metric_graph_edgeattr == "const":
                graph.edge_attr = torch.zeros((num_new_edges, edge_attr_dim))
            # existing edges: edge attr + 0
            # new edges: 0 0 0.. + 1
            elif self.metric_graph_edgeattr == "keep":
                # matrix of 0s
                new_edge_attr = torch.zeros(
                    (graph.num_nodes, graph.num_nodes, edge_attr_dim + 1)
                )
                row, col = graph.edge_index
                # set last bit to 1 for *all* new edges
                new_edge_attr[row, col, -1] = 1

                old_row, old_col = old_edge_ndx
                # existing edges: last bit is 0
                new_edge_attr[old_row, old_col, -1] = 0
                # remaining: insert the edge attr
                new_edge_attr[old_row, old_col, :-1] = graph.edge_attr

                # get the required edge_attr
                graph.edge_attr = new_edge_attr[row, col]

        # get the distance for every edge
        # graph edge index
        # (E, 2) edge index, each row is (i,j) = edge from i to j
        ndx = graph.edge_index.T
        min_edge_lengths = torch.Tensor(min_dist[ndx[:, 0], ndx[:, 1]])
        max_edge_lengths = torch.Tensor(max_dist[ndx[:, 0], ndx[:, 1]])

        # set max lengths
        graph = set_or_append(graph, "distances", torch.Tensor(max_dist))
        graph = set_or_append(graph, "edge_dist", max_edge_lengths)

        graph.max_edge_dist = max_edge_lengths
        graph.min_edge_dist = min_edge_lengths
        graph.max_dist = torch.Tensor(max_dist)
        graph.min_dist = torch.Tensor(min_dist)

        # convert to a basis function representation
        if self.num_dist_basis > 2:
            max_dist_basis = get_dist_basis(
                max_edge_lengths,
                self.dist_basis_type,
                self.num_dist_basis // 2,
                max_dist=4,
            )
            min_dist_basis = get_dist_basis(
                min_edge_lengths,
                self.dist_basis_type,
                self.num_dist_basis // 2,
                max_dist=4,
            )
            joined_basis = torch.cat([max_dist_basis, min_dist_basis], -1)

            graph = set_or_append(graph, "edge_dist_basis", joined_basis)

        return graph
