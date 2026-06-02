from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/compare_science_validation_quality.py"
DATA_DIR = REPO_ROOT / "tests/science.ade4401_data_s1_to_s7"


def load_science_validation_module():
    spec = importlib.util.spec_from_file_location(
        "compare_science_validation_quality_for_tests",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_science_label_mapping_covers_all_55_descriptors() -> None:
    pytest.importorskip("pom_repro")
    module = load_science_validation_module()
    labels = module.read_csv_header(DATA_DIR / "science.ade4401_data_s4.csv")[1:]

    mapping = module.build_label_mapping(labels)

    assert len(labels) == 55
    assert len(mapping) == 55
    assert {item["science_label"] for item in mapping} == set(labels)
    assert any(
        item["science_label"] == "Jasmine" and item["openpom_label"] == "jasmin"
        for item in mapping
    )


def test_available_science_validation_subset_counts() -> None:
    module = load_science_validation_module()

    s1 = module.by_sample(
        module.read_csv_rows(DATA_DIR / "science.ade4401_data_s1.csv")
    )
    s4 = module.by_sample(
        module.read_csv_rows(DATA_DIR / "science.ade4401_data_s4.csv")
    )
    s5 = module.by_sample(
        module.read_csv_rows(DATA_DIR / "science.ade4401_data_s5.csv")
    )
    s7 = module.by_sample(
        module.read_csv_rows(DATA_DIR / "science.ade4401_data_s7.csv")
    )
    s7_with_smiles = {
        sample for sample, row in s7.items() if row.get("SMILES", "").strip()
    }
    clean = {
        sample
        for sample, row in s1.items()
        if not row.get("Disqualification reason", "").strip()
    }
    all_overlap = set(s4) & set(s5) & s7_with_smiles

    assert len(s1) == 400
    assert len(s4) == 397
    assert len(s5) == 397
    assert len(s7_with_smiles) == 209
    assert len(all_overlap) == 207
    assert len(all_overlap & clean) == 163


def test_science_validation_metric_helpers() -> None:
    module = load_science_validation_module()
    left = [1.0, 2.0, 3.0, 4.0]
    right = [1.0, 3.0, 2.0, 4.0]

    assert module.pearson(left, left) == pytest.approx(1.0)
    assert module.spearman(left, left) == pytest.approx(1.0)
    assert module.cosine(left, left) == pytest.approx(1.0)
    assert module.spearman(left, right) == pytest.approx(0.8)

    metrics = module.pair_metrics(left, right)
    assert metrics.pearson == pytest.approx(0.8)
    assert metrics.spearman == pytest.approx(0.8)
    assert metrics.mae == pytest.approx(0.5)
