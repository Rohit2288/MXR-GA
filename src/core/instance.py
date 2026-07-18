"""FSSP instance model: loader + jsonschema validation for the legacy JSONs.

Covers both corpora (task 0.2):
  - Benchmark-FSSP-1: legacy/benchmark-fsspb/instances_out/     (schema 1.0)
  - Benchmark-FSSP-2: legacy/benchmark-fsspb/instances_out_v2/  (schema 1.1)

Instances are canonical and frozen; this module only reads them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import jsonschema

from src.core.fuzzy import TriangularFuzzyNumber

REPO_ROOT = Path(__file__).resolve().parents[2]
FSSP1_DIR = REPO_ROOT / "legacy" / "benchmark-fsspb" / "instances_out"
FSSP2_DIR = REPO_ROOT / "legacy" / "benchmark-fsspb" / "instances_out_v2"

INSTANCE_SCHEMA: dict = {
    "type": "object",
    "required": ["schema_version", "instance_id", "metadata", "items",
                 "target", "witness", "difficulty"],
    "properties": {
        "schema_version": {"enum": ["1.0", "1.1"]},
        "instance_id": {"type": "string", "minLength": 1},
        "metadata": {
            "type": "object",
            "required": ["n", "distribution", "spread_ratio", "asymmetry", "seed"],
            "properties": {
                "n": {"type": "integer", "minimum": 1},
                "distribution": {"type": "string"},
                "spread_ratio": {"type": "number", "exclusiveMinimum": 0},
                "asymmetry": {"type": "number", "exclusiveMinimum": 0},
                "seed": {"type": "integer", "minimum": 0},
            },
        },
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "integer", "minimum": 1},
        },
        "target": {
            "type": "object",
            "required": ["T_L", "T_M", "T_R"],
            "properties": {
                "T_L": {"type": "number"},
                "T_M": {"type": "number"},
                "T_R": {"type": "number"},
            },
        },
        "witness": {
            "type": "object",
            "required": ["subset_indices", "subset_sum", "mu_value"],
            "properties": {
                "subset_indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                },
                "subset_sum": {"type": "integer", "minimum": 0},
                "mu_value": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "difficulty": {"type": "object"},
        "checksum_sha256": {"type": "string"},
    },
}

_VALIDATOR = jsonschema.Draft202012Validator(INSTANCE_SCHEMA)


@dataclass(frozen=True)
class Instance:
    """A single FSSP instance (immutable)."""

    instance_id: str
    items: Tuple[int, ...]
    target: TriangularFuzzyNumber
    n: int
    distribution: str
    spread_ratio: float
    asymmetry: float
    seed: int
    schema_version: str
    suite: Optional[str]          # "T"/"A"/"U" for FSSP-2, None for FSSP-1
    metadata: dict = field(repr=False, default_factory=dict)
    witness: dict = field(repr=False, default_factory=dict)
    difficulty: dict = field(repr=False, default_factory=dict)
    source_path: Optional[Path] = field(repr=False, default=None)


class InstanceValidationError(ValueError):
    """Raised when an instance JSON fails schema or consistency checks."""


def _consistency_checks(raw: dict, path: Optional[Path]) -> None:
    where = str(path) if path else raw.get("instance_id", "<unknown>")
    n = raw["metadata"]["n"]
    if len(raw["items"]) != n:
        raise InstanceValidationError(
            f"{where}: metadata.n={n} but {len(raw['items'])} items")
    t = raw["target"]
    if not (t["T_L"] <= t["T_M"] <= t["T_R"]):
        raise InstanceValidationError(f"{where}: target not ordered: {t}")
    w = raw["witness"]
    idx = w["subset_indices"]
    if len(set(idx)) != len(idx) or any(i >= n for i in idx):
        raise InstanceValidationError(f"{where}: invalid witness indices")
    s = sum(raw["items"][i] for i in idx)
    if s != w["subset_sum"]:
        raise InstanceValidationError(
            f"{where}: witness subset_sum={w['subset_sum']} but items sum to {s}")


def load_instance(path: Path | str) -> Instance:
    """Load, schema-validate, and consistency-check one instance JSON."""
    path = Path(path)
    with open(path) as fh:
        raw = json.load(fh)
    errors = sorted(_VALIDATOR.iter_errors(raw), key=lambda e: e.json_path)
    if errors:
        raise InstanceValidationError(
            f"{path}: schema violations: "
            + "; ".join(e.message for e in errors[:5]))
    _consistency_checks(raw, path)

    md = raw["metadata"]
    suite = md.get("construction", {}).get("v2_suite")
    return Instance(
        instance_id=raw["instance_id"],
        items=tuple(raw["items"]),
        target=TriangularFuzzyNumber.from_dict(raw["target"]),
        n=md["n"],
        distribution=md["distribution"],
        spread_ratio=md["spread_ratio"],
        asymmetry=md["asymmetry"],
        seed=md["seed"],
        schema_version=raw["schema_version"],
        suite=suite,
        metadata=md,
        witness=raw["witness"],
        difficulty=raw["difficulty"],
        source_path=path,
    )


def iter_instance_paths(root: Path) -> Iterator[Path]:
    """Yield all instance JSON paths under a corpus root, sorted for determinism."""
    yield from sorted((root / "instances").rglob("*.json"))


def load_corpus(root: Path) -> List[Instance]:
    """Load and validate every instance under a corpus root."""
    return [load_instance(p) for p in iter_instance_paths(root)]


def load_manifest(root: Path) -> dict:
    """Load the corpus manifest (manifest.json or manifest_v2.json)."""
    for name in ("manifest.json", "manifest_v2.json"):
        p = root / name
        if p.exists():
            with open(p) as fh:
                return json.load(fh)
    raise FileNotFoundError(f"no manifest found under {root}")
