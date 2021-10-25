import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from torch.utils.data import Dataset

import numpy as np

# conversion factors of 12 targets
CONVERSION_FACTOR = torch.Tensor([  1.5034491 ,   8.17294675,   0.59772825,   1.27480012,
            1.2841144 , 280.47258696,   0.90164483,  10.32291767,
            10.41433051,  10.48841883,   9.49758861,   4.06749239
            ])

# means of 12 targets
PYG_MEAN = torch.Tensor([ 2.6821668e+00,  7.5270493e+01, -6.5350609e+00,  3.2208392e-01,
        6.8571439e+00,  1.1894536e+03,  4.0546360e+00, -7.6077148e+01,
       -7.6541321e+01, -7.6978897e+01, -7.0799980e+01,  3.1616858e+01
       ])

def qm9_gnnfilm_to_pyg(data):
    '''
    input: dict with keys 'targets', 'graph', 'id', 'node_features'
    output: Data with edge_attr, edge_index, x, y, z
    '''
    # exclude the float feature in between, keep the rest
    x = torch.Tensor([f[:6] + f[7:] for f in data['node_features']])
    # (1, 12) targets
    y_raw = torch.Tensor(data['targets']).T[:, :-1]
    # convert targets
    y = y_raw * CONVERSION_FACTOR + PYG_MEAN

    # atomic number at 6th position
    z = torch.LongTensor([f[5] for f in data['node_features']])
    # edge types
    dst, edge_types, src = torch.LongTensor(data['graph']).T
    edge_index1 = torch.stack((src, dst))
    edge_index2 = torch.stack((dst, src))
    edge_index = torch.cat((edge_index1, edge_index2), dim=-1)
    # repeat the edge attributes twice
    edge_attr = F.one_hot(edge_types-1, num_classes=4).repeat(2, 1)

    return Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, z=z)


class QM9_GNNFilm(Dataset):
    '''
    Container for a list of Data objects, used to allow a transform on the 
    objects when loaded
    '''
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        data = self.data_list[i]

        if self.transform is not None:
            data = self.transform(data)

        return data

