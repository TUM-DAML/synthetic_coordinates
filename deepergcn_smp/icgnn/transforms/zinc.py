import torch

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdmolops import FastFindRings

class OneHotNodeEdgeFeatures(object):
    def __init__(self, node_types, edge_types):
        self.c = node_types
        self.d = edge_types

    def __call__(self, data):
        n = data.x.shape[0]
        node_encoded = torch.zeros((n, self.c), dtype=torch.float32)
        node_encoded.scatter_(1, data.x.long(), 1)
        data.x = node_encoded
        e = data.edge_attr.shape[0]
        edge_encoded = torch.zeros((e, self.d), dtype=torch.float32)
        edge_attr = (data.edge_attr - 1).long().unsqueeze(-1)
        edge_encoded.scatter_(1, edge_attr, 1)
        data.edge_attr = edge_encoded
        return data

    def __repr__(self):
        return str(self.__class__.__name__)

def zinc_bond_ndx_to_bond(ndx):
    mapping = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    return mapping[ndx]

class ZINC_Reshape_EdgeAttr():
    '''
    Reshape edge attribute from (E,) to (E, 1)
    '''

    def __init__(self):
        pass

    def __call__(self, data):
        data.edge_attr = data.edge_attr.reshape((-1, 1)).float()
        data.y = data.y.reshape((-1, 1)).float()
        return data

class ZINC_Reshape_Target():
    '''
    Reshape edge attribute from (n,) to (n, 1)
    '''

    def __init__(self):
        pass

    def __call__(self, data):
        data.y = data.y.reshape((-1, 1)).float()
        return data

# simple atoms without charge
# map ndx to atomic number
NODE_MAPPING1 = {
    0: 6,       #C
    1: 8,       #O
    2: 7,       #N
    3: 9,       #F
    5: 16,      #S
    6: 17,      #Cl
    9: 35,      #Br
    15: 53,     #I
    16: 15      #P
}

# single atoms with charge
# ndx -> (atomic number, charge)
NODE_MAPPING2 = {
    7: (8, -1),     #O-
    12: (7, 1),     #N+
    13: (7, -1),    #N-
    14: (16, -1),   #S-
    19: (8, 1),     #O+
    20: (16, 1),    #S+
    24: (15, 1)     #P+
}

# radicals/groups with more than 1 atom (heavy atom + hydrogen)
# ndx -> (atomic number, numH, charge)
NODE_MAPPING3 = {
    4: (6, 0, 1),       #CH1
    8: (7, 1, 1),       #NH1+
    10: (7, 3, 1),      #NH3+
    11: (7, 2, 1),      #NH2+
    17: (7, 1, 1),      #OH1+
    18: (7, 1, 1),      #NH1+
    21: (15, 1, 1),     #PH1
    22: (15, 2, 0),     #PH2
    23: (6, 2, -1),     #CH2-
    25: (16, 1, 1),     #SH1+
    26: (6, 1, -1),     #CH1-
    27: (15, 1, 1),     #PH1+
}

def atom_ndx_to_atom(ndx):
    '''
    Create the Chem.Atom corresponding to ndx
    '''
    if ndx in NODE_MAPPING1:
        return Chem.Atom(NODE_MAPPING1[ndx])

    if ndx in NODE_MAPPING2:
        atom_num, charge = NODE_MAPPING2[ndx]
        atom = Chem.Atom(atom_num)
        atom.SetFormalCharge(charge)
        return atom

    if ndx in NODE_MAPPING3:
        atom_num, _, charge = NODE_MAPPING3[ndx]
        atom = Chem.Atom(atom_num)
        atom.SetFormalCharge(charge)
        return atom

    raise ValueError

class ZINC_Graph_To_Mol():
    '''
    Convert ZINC graph to an RDKit molecule
    '''
    def __init__(self):
        pass
    def __call__(self, graph):
        '''
        Map node labels:
        '''
        # create a read/write molecule
        mol = Chem.RWMol()

        add_hs = {}

        # convert one hot encoding to atomic number value
        atom_nums = torch.argmax(graph.x, dim=1).numpy().tolist()

        for atom_ndx, ndx in enumerate(atom_nums):
            # add an atom with this atomic number
            mol.AddAtom(atom_ndx_to_atom(ndx))

            # check if its heavy atom+H
            if ndx in NODE_MAPPING3:
                _, num_h, _ = NODE_MAPPING3[ndx]
                # store the number of Hs to be added
                add_hs[atom_ndx] = num_h
        
        # where to start indexing new H atoms
        h_ndx = len(graph.x)

        for atom_ndx, num_hs in add_hs.items():
            for _ in range(num_hs):
                mol.AddAtom(Chem.Atom(1))
                # bond from atom to this H
                mol.AddBond(atom_ndx, h_ndx, Chem.BondType.SINGLE)
                # increment every time a H is added
                h_ndx += 1

        '''
        Map Edge labels:
        'SINGLE': 1
        'DOUBLE': 2
        'TRIPLE': 3
        '''
        # bond type for each bond - single, double, triple
        bond_vals = (torch.argmax(graph.edge_attr, dim=1) + 1).numpy()
        bonds = list(map(zinc_bond_ndx_to_bond, bond_vals))    

        # create a bond, set its properties and add to the molecule
        for ndx, (i, j) in enumerate(graph.edge_index.T):
            bond = mol.GetBondBetweenAtoms(i.item(), j.item()) 
            if bond is None:
                mol.AddBond(i.item(), j.item(), bonds[ndx]) 

        # cleanup the molecule
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache()
        FastFindRings(mol)

        return (graph, mol)