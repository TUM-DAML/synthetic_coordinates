import torch

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdmolops import FastFindRings


class RemoveTargets:
    def __init__(self, remove_ndx=None, keep_ndx=None):
        """
        Specify any one:
            keep_ndx (tuple): indices of targets to keep
            remove_ndx (tuple): indices of targets to remove
        """
        if remove_ndx:
            self.keep_ndx = [ndx for ndx in range(19) if ndx not in remove_ndx]
        else:
            self.keep_ndx = keep_ndx

        print(f"QM9: Keep indices {self.keep_ndx}")

    def __call__(self, data):
        data.y = data.y[:, self.keep_ndx]

        return data


def bond_ndx_to_bond(ndx):
    mapping = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC,
    }
    return mapping[ndx]


class QM9_Graph_To_Mol:
    def __init__(self):
        pass

    def __call__(self, graph):
        # create a read/write molecule
        mol = Chem.RWMol()

        for atom_num in graph.z:
            # add an atom with this atomic number
            mol.AddAtom(Chem.Atom(atom_num.item()))

        bond_types_int = torch.argmax(graph.edge_attr, dim=1)
        bond_types = list(map(bond_ndx_to_bond, bond_types_int.numpy()))

        num_edges = graph.edge_index.shape[1]

        # each edge added twice, keep only the first half
        real_edges = graph.edge_index.T[: int(num_edges / 2), :]

        for ndx, (i, j) in enumerate(real_edges):
            try:
                mol.AddBond(i.item(), j.item(), bond_types[ndx])
            # TODO: pick the set of edges correctly based on the permutation?
            except RuntimeError:
                pass

        # cleanup the molecule
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache()
        FastFindRings(mol)

        return graph, mol
