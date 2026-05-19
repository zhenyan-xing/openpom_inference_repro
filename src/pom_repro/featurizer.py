"""OpenPOM-compatible RDKit graph featurization.

This module intentionally mirrors the feature order and SMILES handling used by
OpenPOM's ``openpom.feat.graph_featurizer.GraphFeaturizer`` while avoiding a
runtime dependency on OpenPOM or DeepChem.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops

logger = logging.getLogger(__name__)


class GraphConvConstants:
    """Feature dimensions and allowable values used by OpenPOM."""

    MAX_ATOMIC_NUM = 100
    ATOM_FEATURES: dict[str, list[int]] = {
        "valence": [0, 1, 2, 3, 4, 5, 6],
        "degree": [0, 1, 2, 3, 4, 5],
        "num_Hs": [0, 1, 2, 3, 4],
        "formal_charge": [-1, -2, 1, 2, 0],
        "atomic_num": list(range(MAX_ATOMIC_NUM)),
    }
    ATOM_FEATURES_HYBRIDIZATION: list[str] = [
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
    ]
    ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + (
        len(ATOM_FEATURES_HYBRIDIZATION) + 1
    )
    BOND_FDIM = 6


def _one_hot_encode(
    value: int | str,
    allowable_set: Sequence[int] | Sequence[str],
    include_unknown_set: bool = False,
) -> list[float]:
    if include_unknown_set:
        one_hot = [0.0] * (len(allowable_set) + 1)
    else:
        one_hot = [0.0] * len(allowable_set)

    try:
        one_hot[allowable_set.index(value)] = 1.0  # type: ignore[arg-type]
    except ValueError:
        if include_unknown_set:
            one_hot[-1] = 1.0
    return one_hot


def _get_atomic_num_one_hot(atom: Chem.rdchem.Atom) -> list[float]:
    return _one_hot_encode(
        atom.GetAtomicNum() - 1,
        GraphConvConstants.ATOM_FEATURES["atomic_num"],
        include_unknown_set=True,
    )


def _get_atom_total_valence_one_hot(atom: Chem.rdchem.Atom) -> list[float]:
    return _one_hot_encode(
        atom.GetTotalValence(),
        GraphConvConstants.ATOM_FEATURES["valence"],
        include_unknown_set=True,
    )


def _get_atom_total_degree_one_hot(atom: Chem.rdchem.Atom) -> list[float]:
    return _one_hot_encode(
        atom.GetTotalDegree(),
        GraphConvConstants.ATOM_FEATURES["degree"],
        include_unknown_set=True,
    )


def _get_atom_total_num_hs_one_hot(atom: Chem.rdchem.Atom) -> list[float]:
    return _one_hot_encode(
        atom.GetTotalNumHs(),
        GraphConvConstants.ATOM_FEATURES["num_Hs"],
        include_unknown_set=True,
    )


def _get_atom_formal_charge_one_hot(atom: Chem.rdchem.Atom) -> list[float]:
    return _one_hot_encode(
        atom.GetFormalCharge(),
        GraphConvConstants.ATOM_FEATURES["formal_charge"],
        include_unknown_set=True,
    )


def _get_atom_hybridization_one_hot(atom: Chem.rdchem.Atom) -> list[float]:
    return _one_hot_encode(
        str(atom.GetHybridization()),
        GraphConvConstants.ATOM_FEATURES_HYBRIDIZATION,
        include_unknown_set=True,
    )


def atom_features(atom: Chem.rdchem.Atom | None) -> Sequence[int | float]:
    """Return the 134-dimensional OpenPOM atom feature vector."""

    if atom is None:
        return [0] * GraphConvConstants.ATOM_FDIM

    features: list[float] = []
    features += _get_atom_total_valence_one_hot(atom)
    features += _get_atom_total_degree_one_hot(atom)
    features += _get_atom_total_num_hs_one_hot(atom)
    features += _get_atom_formal_charge_one_hot(atom)
    features += _get_atomic_num_one_hot(atom)
    features += _get_atom_hybridization_one_hot(atom)
    return [int(feature) for feature in features]


def bond_features(bond: Chem.rdchem.Bond | None) -> Sequence[int | bool]:
    """Return the 6-dimensional OpenPOM bond feature vector."""

    if bond is None:
        return [1] + [0] * (GraphConvConstants.BOND_FDIM - 1)

    bond_type = bond.GetBondType()
    return [
        0,
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.IsInRing(),
    ]


class GraphData:
    """Small DeepChem-compatible graph container used by OpenPOM inputs."""

    def __init__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_features: np.ndarray | None = None,
        node_pos_features: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(node_features, np.ndarray):
            raise ValueError("node_features must be np.ndarray.")
        if not isinstance(edge_index, np.ndarray):
            raise ValueError("edge_index must be np.ndarray.")
        if not np.issubdtype(edge_index.dtype, np.integer):
            raise ValueError("edge_index.dtype must contain integers.")
        if edge_index.shape[0] != 2:
            raise ValueError("The shape of edge_index is [2, num_edges].")
        if edge_index.size and int(np.max(edge_index)) >= len(node_features):
            raise ValueError("edge_index contains the invalid node number.")

        if edge_features is not None:
            if not isinstance(edge_features, np.ndarray):
                raise ValueError("edge_features must be np.ndarray or None.")
            if edge_index.shape[1] != edge_features.shape[0]:
                raise ValueError(
                    "The first dimension of edge_features must match edge_index."
                )

        if node_pos_features is not None:
            if not isinstance(node_pos_features, np.ndarray):
                raise ValueError("node_pos_features must be np.ndarray or None.")
            if node_pos_features.shape[0] != node_features.shape[0]:
                raise ValueError(
                    "node_pos_features length must match node_features length."
                )

        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.node_pos_features = node_pos_features
        self.kwargs = kwargs
        self.num_nodes, self.num_node_features = self.node_features.shape
        self.num_edges = self.edge_index.shape[1]
        if self.edge_features is not None:
            self.num_edge_features = self.edge_features.shape[1]

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        edge_features = (
            "None"
            if self.edge_features is None
            else str(list(self.edge_features.shape))
        )
        return (
            f"GraphData(node_features={list(self.node_features.shape)}, "
            f"edge_index={list(self.edge_index.shape)}, "
            f"edge_features={edge_features})"
        )

    def to_dgl_graph(self, self_loop: bool = False):
        """Convert this graph to a DGL graph with OpenPOM field names."""

        try:
            import dgl
            import torch
        except ModuleNotFoundError as exc:
            raise ImportError("This function requires DGL and PyTorch.") from exc

        src = self.edge_index[0]
        dst = self.edge_index[1]
        graph = dgl.graph(
            (torch.from_numpy(src).long(), torch.from_numpy(dst).long()),
            num_nodes=self.num_nodes,
        )
        graph.ndata["x"] = torch.from_numpy(self.node_features).float()

        if self.node_pos_features is not None:
            graph.ndata["pos"] = torch.from_numpy(self.node_pos_features).float()
            graph.edata["d"] = torch.norm(
                graph.ndata["pos"][graph.edges()[0]]
                - graph.ndata["pos"][graph.edges()[1]],
                p=2,
                dim=-1,
            ).unsqueeze(-1).detach()

        if self.edge_features is not None:
            graph.edata["edge_attr"] = torch.from_numpy(self.edge_features).float()

        if self_loop:
            graph.add_edges(np.arange(self.num_nodes), np.arange(self.num_nodes))

        return graph


class GraphFeaturizer:
    """OpenPOM-compatible featurizer for SMILES strings and RDKit molecules."""

    def __init__(self, is_adding_hs: bool = False) -> None:
        self.is_adding_hs = is_adding_hs
        self.use_original_atoms_order = False

    def _construct_bond_index(self, datapoint: Chem.rdchem.Mol) -> np.ndarray:
        src: list[int] = []
        dest: list[int] = []
        for bond in datapoint.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]
        return np.asarray([src, dest], dtype=int)

    def _featurize(self, datapoint: Chem.rdchem.Mol, **kwargs: Any) -> GraphData:
        if isinstance(datapoint, Chem.rdchem.Mol):
            if self.is_adding_hs:
                datapoint = Chem.AddHs(datapoint)
        else:
            raise ValueError("Feature field should contain smiles for featurizer!")

        node_features = np.asarray(
            [atom_features(atom) for atom in datapoint.GetAtoms()],
            dtype=float,
        )

        if len(datapoint.GetBonds()) == 0:
            edge_features = np.empty((0, GraphConvConstants.BOND_FDIM))
        else:
            edge_feature_rows = []
            for bond in datapoint.GetBonds():
                edge_feature_rows.extend(2 * [bond_features(bond)])
            edge_features = np.asarray(edge_feature_rows, dtype=float)

        edge_index = self._construct_bond_index(datapoint)
        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )

    def _mol_from_smiles(self, smiles: str) -> Chem.rdchem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        if self.use_original_atoms_order:
            return mol

        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        return rdmolops.RenumberAtoms(mol, new_order)

    def featurize(
        self,
        datapoints: str | Chem.rdchem.Mol | Sequence[str | Chem.rdchem.Mol],
        log_every_n: int = 1000,
        **kwargs: Any,
    ) -> np.ndarray:
        """Calculate OpenPOM-style graph features for one or more molecules."""

        if isinstance(datapoints, str) or isinstance(datapoints, Chem.rdchem.Mol):
            molecules = [datapoints]
        else:
            molecules = list(datapoints)

        features: list[GraphData | np.ndarray] = []
        for index, mol in enumerate(molecules):
            if index % log_every_n == 0:
                logger.info("Featurizing datapoint %i", index)

            try:
                if isinstance(mol, str):
                    mol = self._mol_from_smiles(mol)

                kwargs_per_datapoint = {
                    key: value[index] for key, value in kwargs.items()
                }
                features.append(self._featurize(mol, **kwargs_per_datapoint))
            except Exception as exc:  # match DeepChem featurizer failure behavior
                mol_name = (
                    Chem.MolToSmiles(mol)
                    if isinstance(mol, Chem.rdchem.Mol)
                    else mol
                )
                logger.warning(
                    "Failed to featurize datapoint %d, %s. Appending empty array",
                    index,
                    mol_name,
                )
                logger.warning("Exception message: %s", exc)
                features.append(np.array([]))

        try:
            return np.asarray(features)
        except ValueError as exc:
            logger.warning("Exception message: %s", exc)
            return np.asarray(features, dtype=object)
