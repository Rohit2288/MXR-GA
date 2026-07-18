"""Tests for the instance loader and validation (task 0.2)."""

import json

import pytest

from src.core.instance import (FSSP1_DIR, FSSP2_DIR, InstanceValidationError,
                               iter_instance_paths, load_instance)


def first_path(root):
    return next(iter(iter_instance_paths(root)))


class TestLoad:
    def test_loads_fssp1_instance(self):
        inst = load_instance(first_path(FSSP1_DIR))
        assert inst.n == len(inst.items)
        assert inst.suite is None
        assert inst.schema_version == "1.0"
        assert inst.target.T_L <= inst.target.T_M <= inst.target.T_R

    def test_loads_fssp2_instance(self):
        inst = load_instance(first_path(FSSP2_DIR))
        assert inst.suite in {"T", "A", "U"}
        assert inst.schema_version == "1.1"

    def test_witness_sum_consistent(self):
        inst = load_instance(first_path(FSSP2_DIR))
        s = sum(inst.items[i] for i in inst.witness["subset_indices"])
        assert s == inst.witness["subset_sum"]


class TestValidation:
    @pytest.fixture
    def raw(self):
        with open(first_path(FSSP2_DIR)) as fh:
            return json.load(fh)

    def _write(self, tmp_path, raw):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(raw))
        return p

    def test_missing_field_rejected(self, tmp_path, raw):
        del raw["target"]
        with pytest.raises(InstanceValidationError):
            load_instance(self._write(tmp_path, raw))

    def test_item_count_mismatch_rejected(self, tmp_path, raw):
        raw["metadata"]["n"] = len(raw["items"]) + 1
        with pytest.raises(InstanceValidationError):
            load_instance(self._write(tmp_path, raw))

    def test_unordered_target_rejected(self, tmp_path, raw):
        raw["target"]["T_L"] = raw["target"]["T_R"] + 1
        with pytest.raises(InstanceValidationError):
            load_instance(self._write(tmp_path, raw))

    def test_bad_witness_sum_rejected(self, tmp_path, raw):
        raw["witness"]["subset_sum"] += 1
        with pytest.raises(InstanceValidationError):
            load_instance(self._write(tmp_path, raw))
