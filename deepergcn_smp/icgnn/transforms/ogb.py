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
