#!/usr/bin/env python
"""Compare local graph features against the original OpenPOM featurizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pom_repro.featurizer import GraphFeaturizer  # noqa: E402


DEFAULT_OPENPOM_ROOT = Path("/home/xing/openpom")
DEFAULT_SMILES = [
    "C",
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(C)O",
    "C1=CC=CC=C1O",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pom_repro graph features with original OpenPOM."
    )
    parser.add_argument(
        "--openpom-root",
        type=Path,
        default=DEFAULT_OPENPOM_ROOT,
        help=f"Read-only OpenPOM source tree. Default: {DEFAULT_OPENPOM_ROOT}",
    )
    parser.add_argument(
        "--smiles",
        action="append",
        default=None,
        help="SMILES to compare. Repeat to override the default list.",
    )
    parser.add_argument(
        "--smiles-file",
        type=Path,
        default=None,
        help="Optional newline-delimited SMILES file. Overrides defaults.",
    )
    return parser.parse_args()


def load_smiles(args: argparse.Namespace) -> list[str]:
    if args.smiles_file is not None:
        lines = args.smiles_file.read_text(encoding="utf-8").splitlines()
        smiles = [line.strip() for line in lines if line.strip()]
    elif args.smiles is not None:
        smiles = args.smiles
    else:
        smiles = DEFAULT_SMILES

    if not smiles:
        raise ValueError("SMILES list is empty.")
    return smiles


def import_openpom_featurizer(openpom_root: Path) -> type[Any]:
    if not openpom_root.exists():
        raise FileNotFoundError(f"OpenPOM root does not exist: {openpom_root}")
    sys.path.insert(0, str(openpom_root))
    from openpom.feat.graph_featurizer import (
        GraphFeaturizer as OpenPOMGraphFeaturizer,
    )

    return OpenPOMGraphFeaturizer


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def graph_arrays(graph: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(graph.node_features),
        np.asarray(graph.edge_features),
        np.asarray(graph.edge_index),
    )


def compare_graphs(smiles: list[str], openpom_featurizer: Any) -> bool:
    local_graphs = GraphFeaturizer().featurize(smiles)
    openpom_graphs = openpom_featurizer.featurize(smiles)

    all_ok = True
    print("Comparing graph features")
    print(f"num_molecules = {len(smiles)}")

    for smile, local_graph, openpom_graph in zip(
        smiles, local_graphs, openpom_graphs, strict=True
    ):
        local_nodes, local_edges, local_edge_index = graph_arrays(local_graph)
        ref_nodes, ref_edges, ref_edge_index = graph_arrays(openpom_graph)

        atom_diff = max_abs_diff(local_nodes, ref_nodes)
        bond_diff = max_abs_diff(local_edges, ref_edges)
        edge_order_equal = np.array_equal(local_edge_index, ref_edge_index)
        shapes_equal = (
            local_nodes.shape == ref_nodes.shape
            and local_edges.shape == ref_edges.shape
            and local_edge_index.shape == ref_edge_index.shape
        )
        ok = atom_diff == 0 and bond_diff == 0 and edge_order_equal and shapes_equal
        all_ok = all_ok and ok

        print(f"\nSMILES: {smile}")
        print(f"  node shape local/ref: {local_nodes.shape} / {ref_nodes.shape}")
        print(f"  edge shape local/ref: {local_edges.shape} / {ref_edges.shape}")
        print(
            "  edge_index shape local/ref: "
            f"{local_edge_index.shape} / {ref_edge_index.shape}"
        )
        print(f"  atom feature max_abs_diff = {atom_diff:g}")
        print(f"  bond feature max_abs_diff = {bond_diff:g}")
        print(f"  edge order identical = {edge_order_equal}")
        print(f"  result = {'PASS' if ok else 'FAIL'}")

    print("\nOverall result:", "PASS" if all_ok else "FAIL")
    return all_ok


def main() -> int:
    args = parse_args()
    smiles = load_smiles(args)
    openpom_featurizer_cls = import_openpom_featurizer(args.openpom_root.resolve())
    ok = compare_graphs(smiles, openpom_featurizer_cls())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
