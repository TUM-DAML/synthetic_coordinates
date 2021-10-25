import numpy as np

from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdmolops import FastFindRings


def bond_ndx_to_bond(ndx):
    mapping = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC,
    }
    return mapping[ndx]

def qm9_to_rdkit(mol):
    '''
    mol: molecule dict from the dataset
    return: corresponding RDKit molecule
    '''
    rdk_mol = Chem.RWMol()
    atom_nums = [feat[5] for feat in mol['node_features']]

    for atom_num in atom_nums:
        # add an atom with this atomic number
        rdk_mol.AddAtom(Chem.Atom(atom_num))
    
    for _, (i, type, j) in enumerate(mol['graph']):
        rdk_mol.AddBond(j, i, bond_ndx_to_bond(type - 1)) 

    # cleanup the molecule
    Chem.SanitizeMol(rdk_mol)
    rdk_mol.UpdatePropertyCache()
    FastFindRings(rdk_mol)

    return rdk_mol

def get_upper_tri_distances(bmat):
    '''
    Use the upper triangular part of the bounds matrix
    to get an all-pairs distances matrix 
    by reflecting the upper tri part about the diagonal

    bmat: bounds matrix from RDKit
    '''
    # use only the upper bounds
    # reflect it across the diagonal
    upper = np.triu(bmat)
    lower = upper.T

    distances = lower + upper

    return distances

def get_dist_bounds(mol):
    '''
    mol: RDKit molecule
    return: (N, N) matrix of distances
    '''

    bm = molDG.GetMoleculeBoundsMatrix(mol)
    max_dist = get_upper_tri_distances(bm)
    min_dist = get_upper_tri_distances(bm.T)

    return min_dist, max_dist