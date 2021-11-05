import rdkit

import numpy as np
import scipy.sparse as sp
from .utils_gnn_film import read_jsonl

from .utils_rdkit import qm9_to_rdkit, get_dist_bounds

from tqdm import tqdm

index_keys = ["batch_seg", "idnb_i", "idnb_j", "id_expand_kj", "id_reduce_ji"]


class DataContainer:
    def __init__(
        self, filename, target_keys, dist={"type": "none"}, subset=False, ablation=None
    ):
        # option for const angle, const dist
        self.ablation = ablation
        print(f"Use ablation: {self.ablation}")
        # read the GNN FILM dataset
        self.data = list(read_jsonl(filename))
        if subset:
            print("Selecting a subset of data")
            self.data = self.data[:16]

        targets = [
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "U0",
            "U",
            "H",
            "G",
            "Cv",
            "omega",
        ]
        target_idx = [targets.index(key) for key in target_keys]

        # mean of pyg QM9 19 targets
        pyg_qm9_mean = np.array(
            [
                2.6821668e00,
                7.5270493e01,
                -6.5350609e00,
                3.2208392e-01,
                6.8571439e00,
                1.1894536e03,
                4.0546360e00,
                -1.1182017e04,
                -1.1181785e04,
                -1.1181761e04,
                -1.1182925e04,
                3.1616858e01,
                -7.6077148e01,
                -7.6541321e01,
                -7.6978897e01,
                -7.0799980e01,
                9.9633703e00,
                1.4066898e00,
                1.1270893e00,
            ]
        )

        # 12 indices in the pyg dataset corresponding to the targets in the FILM dataset
        pyg_idx_map = [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11]

        # conversion factors for 13 GNN FILM targets - dont need omega (last one)
        conversion_all = np.array(
            [
                1.5034491,
                8.17294675,
                0.59772825,
                1.27480012,
                1.2841144,
                280.47258696,
                0.90164483,
                10.32291767,
                10.41433051,
                10.48841883,
                9.49758861,
                4.06749239,
                266.89827743,
            ]
        )

        conversion = conversion_all[target_idx]

        if dist["type"] in ["none", "ppr", "rdkit_bounds", "ppr_rdkit_bounds"]:
            self.dist = dist
        else:
            raise ValueError(f"Unknown distance type: '{dist['type']}'")

        # get the indices into the pyg targets
        pyg_idx = [pyg_idx_map[i] for i in target_idx]
        # pick only the required target means
        targets_mean = pyg_qm9_mean[pyg_idx]

        data_new = []
        skipped = 0

        print("Processing molecules")
        for _, mol in enumerate(tqdm(self.data)):
            # convert target from GNN film to Pytorch geometric (standard) units
            converted_target = (
                np.array(mol["targets"])[target_idx] * conversion + targets_mean
            )
            mol["targets"] = converted_target.flatten()
            mol["adj_matrix"], mol["edge_type"] = self._edge_list_to_csr(mol["graph"])
            # Calculate distances
            if self.dist["type"] == "none":
                pass
            if self.dist["type"] in ("ppr", "ppr_rdkit_bounds"):
                ppr = self._get_ppr_matrix(mol["adj_matrix"], self.dist["alpha"])
                Dij = self._sim_to_dist(ppr)
                mol["dists"] = sp.csr_matrix(Dij)
            if self.dist["type"] in ("rdkit_bounds", "ppr_rdkit_bounds"):
                try:
                    mol = self._set_rdkit_bounds(mol)
                    mol["min_dist"] = sp.csr_matrix(mol["min_dist"])
                    mol["max_dist"] = sp.csr_matrix(mol["max_dist"])
                except rdkit.Chem.rdchem.AtomValenceException:
                    skipped += 1
                    continue
            data_new.append(mol)

        print(f"Skipped: {skipped}")
        self.data = data_new
        self.N = np.array([len(g["node_features"]) for g in self.data])

    def _set_rdkit_bounds(self, mol):
        """
        Get the RDKit bounds matrix
        mol: the molecule dictionary

        return:
            dist: (N, N) array
        """
        rdkit_mol = qm9_to_rdkit(mol)
        min_dist, max_dist = get_dist_bounds(rdkit_mol)

        mol["min_dist"], mol["max_dist"] = min_dist, max_dist

        return mol

    def _get_ppr_matrix(self, adj_matrix, alpha, eps=1e-6):
        natoms = adj_matrix.shape[0]
        deg = adj_matrix.sum(1).A1
        deg_inv_sqrt = np.sqrt(1 / (deg + eps))
        T_sym = deg_inv_sqrt[None, :] * adj_matrix.A * deg_inv_sqrt[:, None]
        ppr = alpha * np.linalg.inv(np.eye(natoms) - (1 - alpha) * T_sym)
        return ppr

    def _sim_to_dist(self, sim):
        diag = np.diag(sim)
        dist2 = diag[:, None] + diag[None, :] - 2 * sim
        dist = np.sqrt(np.maximum(dist2, 0))
        return dist

    def _edge_list_to_csr(self, edge_list):
        edge_array = np.array(edge_list)
        edgeid_to_target, edge_type, edgeid_to_source = edge_array.T
        natoms = np.max([edgeid_to_target, edgeid_to_source]) + 1
        adj_matrix = sp.csr_matrix(
            (edge_type, (edgeid_to_target, edgeid_to_source)), (natoms, natoms)
        )
        adj_matrix += adj_matrix.T  # Make adjacency symmetric
        edge_type = adj_matrix.data - 1  # Make edge type start from 0
        adj_matrix.data.fill(1)
        return adj_matrix, edge_type

    def _bmat_fast(self, mats):
        new_data = np.concatenate([mat.data for mat in mats])

        ind_offset = np.zeros(1 + len(mats))
        ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
        new_indices = np.concatenate(
            [mats[i].indices + ind_offset[i] for i in range(len(mats))]
        )

        indptr_offset = np.zeros(1 + len(mats))
        indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
        new_indptr = np.concatenate(
            [mats[i].indptr[i >= 1 :] + indptr_offset[i] for i in range(len(mats))]
        )
        return sp.csr_matrix((new_data, new_indices, new_indptr))

    def _calculate_neighbor_angles_diff(
        self, dij, djk, dik, id3_i, id3_j, id3_k, eps=1e-6
    ):
        """Calculate angles for neighboring atom triplets
        where distances for each edge can come from different matrices
        dij: matrix of distances for ij
        djk: matrix of distances for jk
        dik: matrix of distances for ik

        """
        dij2 = dij.power(2)
        djk2 = dij.power(2)
        dik2 = dik.power(2)

        D2ij = dij2[id3_i, id3_j].A1
        D2ik = dik2[id3_i, id3_k].A1
        D2jk = djk2[id3_j, id3_k].A1

        Dij = dij[id3_i, id3_j].A1
        Djk = djk[id3_j, id3_k].A1

        angle = np.arccos(
            np.clip((D2ij + D2jk - D2ik) / (2 * Dij * Djk + eps), a_min=-1, a_max=1)
        )

        # Subtract from π since one vector is incoming, the other outgoing.
        # This is just convention, it shouldn't affect anything
        # since it just flips some signs in the basis.
        angle = np.pi - angle

        return angle

    def _calculate_neighbor_angles(self, dists, id3_i, id3_j, id3_k):
        """Calculate angles for neighboring atom triplets"""
        dists2 = dists.power(2)
        D2ij = dists2[id3_i, id3_j].A1
        D2ik = dists2[id3_i, id3_k].A1
        D2jk = dists2[id3_j, id3_k].A1
        Dij = dists[id3_i, id3_j].A1
        Djk = dists[id3_j, id3_k].A1
        angle = np.arccos(
            np.clip((D2ij + D2jk - D2ik) / (2 * Dij * Djk), a_min=-1, a_max=1)
        )

        # Subtract from π since one vector is incoming, the other outgoing.
        # This is just convention, it shouldn't affect anything
        # since it just flips some signs in the basis.
        angle = np.pi - angle

        return angle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mols = [self.data[i] for i in idx]

        data = {}
        data["targets"] = np.vstack([mol["targets"] for mol in mols])

        node_attr = []
        for mol in mols:
            mol_attr = [f[:6] + f[7:] for f in mol["node_features"]]
            node_attr.append(mol_attr)
        data["node_attr"] = np.vstack(node_attr)

        data["edge_type"] = np.hstack([mol["edge_type"] for mol in mols])

        data["batch_seg"] = np.repeat(np.arange(len(idx), dtype=np.int32), self.N[idx])

        # Entry x,y is edge x<-y (!)
        adj_matrix = self._bmat_fast([mol["adj_matrix"] for mol in mols])
        # Entry x,y is edgeid x<-y (!)
        atomids_to_edgeid = sp.csr_matrix(
            (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
            shape=adj_matrix.shape,
        )
        edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()

        # Target (i) and source (j) nodes of edges
        data["idnb_i"] = edgeid_to_target
        data["idnb_j"] = edgeid_to_source

        # Indices of triplets k->j->i
        ntriplets = adj_matrix[edgeid_to_source].sum(1).A1
        id3ynb_i = np.repeat(edgeid_to_target, ntriplets)
        id3ynb_j = np.repeat(edgeid_to_source, ntriplets)
        id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]

        # Indices of triplets that are not i->j->i
        (id3_y_to_d,) = (id3ynb_i != id3ynb_k).nonzero()
        id3dnb_i = id3ynb_i[id3_y_to_d]
        id3dnb_j = id3ynb_j[id3_y_to_d]
        id3dnb_k = id3ynb_k[id3_y_to_d]

        # Edge indices for interactions
        # j->i => k->j
        data["id_expand_kj"] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
        # j->i => k->j => j->i
        data["id_reduce_ji"] = (
            atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]
        )

        Dij, Anglesijk = [], []

        # store distances and angles
        if self.dist["type"] in ("ppr", "ppr_rdkit_bounds"):
            dists = self._bmat_fast([mol["dists"] for mol in mols])
            Dij.extend([dists[edgeid_to_target, edgeid_to_source].A1])
            Anglesijk.extend(
                [self._calculate_neighbor_angles(dists, id3dnb_i, id3dnb_j, id3dnb_k)]
            )

        if self.dist["type"] in ("rdkit_bounds", "ppr_rdkit_bounds"):
            min_dists = self._bmat_fast([mol["min_dist"] for mol in mols])
            max_dists = self._bmat_fast([mol["max_dist"] for mol in mols])
            mean_dists = self._bmat_fast(
                [(mol["min_dist"] + mol["max_dist"]) / 2 for mol in mols]
            )

            dij_min = min_dists[edgeid_to_target, edgeid_to_source].A1
            dij_max = max_dists[edgeid_to_target, edgeid_to_source].A1
            # add mean distance as well, to be multiplied with mean angle later
            dij_mean = mean_dists[edgeid_to_target, edgeid_to_source].A1

            Dij.extend([dij_min, dij_max, dij_mean])

            angle_min = self._calculate_neighbor_angles_diff(
                max_dists, max_dists, min_dists, id3dnb_i, id3dnb_j, id3dnb_k
            )
            angle_max = self._calculate_neighbor_angles_diff(
                min_dists, min_dists, max_dists, id3dnb_i, id3dnb_j, id3dnb_k
            )
            angle_center = self._calculate_neighbor_angles_diff(
                mean_dists, mean_dists, mean_dists, id3dnb_i, id3dnb_j, id3dnb_k
            )
            Anglesijk.extend([angle_min, angle_max, angle_center])

        # multiple distances and angles - combine them and add a new dimension
        if len(Dij) > 1 and len(Anglesijk) > 1:
            data["Dij"] = np.stack(Dij, axis=-1)
            data["Anglesijk"] = np.stack(Anglesijk, axis=-1)
        # only one distance and angle - dont add a new dimension
        else:
            data["Dij"] = Dij[0]
            data["Anglesijk"] = Anglesijk[0]

        # set angle to constant values
        if self.ablation in (
            "const_angle",
            "const_both",
        ):
            data["Anglesijk"] = np.ones_like(data["Anglesijk"])
        if self.ablation in ("const_both",):
            data["Dij"] = np.ones_like(data["Dij"])

        return data
