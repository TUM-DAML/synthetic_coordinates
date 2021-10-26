from rdkit.Chem import AllChem as Chem


def get_chiral_tag(ndx):
    mapping = {
        0: Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        1: Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        2: Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        3: Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        4: Chem.rdchem.ChiralType.CHI_OTHER,
    }
    return mapping[ndx]


def get_formal_charge(ndx):
    charges = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    mapping = dict(zip(range(len(charges)), charges))
    return mapping[ndx]


def get_hybridization(ndx):
    mapping = {
        0: Chem.rdchem.HybridizationType.SP,
        1: Chem.rdchem.HybridizationType.SP2,
        2: Chem.rdchem.HybridizationType.SP3,
        3: Chem.rdchem.HybridizationType.SP3D,
        4: Chem.rdchem.HybridizationType.SP3D2,
        5: Chem.rdchem.HybridizationType.OTHER,
    }
    return mapping[ndx]


def get_aromatic(ndx):
    mapping = {0: False, 1: True}
    return mapping[ndx]


def bond_ndx_to_bond(ndx):
    mapping = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC,
    }
    return mapping[ndx]


def get_bond_stereo(ndx):
    mapping = {
        0: Chem.rdchem.BondStereo.STEREONONE,
        1: Chem.rdchem.BondStereo.STEREOZ,
        2: Chem.rdchem.BondStereo.STEREOE,
        3: Chem.rdchem.BondStereo.STEREOCIS,
        4: Chem.rdchem.BondStereo.STEREOTRANS,
        5: Chem.rdchem.BondStereo.STEREOANY,
    }
    return mapping[ndx]


def get_conjugated(ndx):
    mapping = {0: False, 1: True}
    return mapping[ndx]
