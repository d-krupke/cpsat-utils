"""
Tests for cpsat_utils.io (export_model, import_model).
"""

import pytest
from ortools.sat.python import cp_model

from cpsat_utils.io import export_model, import_model
from cpsat_utils.testing import assert_objective


class TestExportImportBinary:
    def test_roundtrip_binary(self, tmp_path):
        model = cp_model.CpModel()
        x = model.new_int_var(0, 10, "x")
        y = model.new_int_var(0, 10, "y")
        model.add(x + y == 7)
        model.minimize(x)

        filepath = str(tmp_path / "model.pb")
        export_model(model, filepath)
        loaded = import_model(filepath)
        assert_objective(loaded, expected=0.0)

    def test_roundtrip_bin_extension(self, tmp_path):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        model.minimize(x)

        filepath = str(tmp_path / "model.bin")
        export_model(model, filepath)
        loaded = import_model(filepath)
        assert_objective(loaded, expected=1.0)

    def test_roundtrip_dat_extension(self, tmp_path):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        model.minimize(x)

        filepath = str(tmp_path / "model.dat")
        export_model(model, filepath)
        loaded = import_model(filepath)
        assert_objective(loaded, expected=1.0)


class TestExportImportText:
    def test_roundtrip_text_pbtxt(self, tmp_path):
        model = cp_model.CpModel()
        x = model.new_int_var(0, 10, "x")
        y = model.new_int_var(0, 10, "y")
        model.add(x + y == 7)
        model.minimize(x)

        filepath = str(tmp_path / "model.pbtxt")
        export_model(model, filepath)
        loaded = import_model(filepath)
        assert_objective(loaded, expected=0.0)

    def test_roundtrip_text_txt(self, tmp_path):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        model.minimize(x)

        filepath = str(tmp_path / "model.txt")
        export_model(model, filepath)
        loaded = import_model(filepath)
        assert_objective(loaded, expected=1.0)

    def test_text_file_is_human_readable(self, tmp_path):
        model = cp_model.CpModel()
        model.new_int_var(0, 10, "my_var")

        filepath = str(tmp_path / "model.pbtxt")
        export_model(model, filepath)
        content = (tmp_path / "model.pbtxt").read_text()
        assert "my_var" in content


class TestEdgeCases:
    def test_unknown_extension_raises(self, tmp_path):
        model = cp_model.CpModel()
        with pytest.raises(ValueError, match="Cannot determine format"):
            export_model(model, str(tmp_path / "model.xyz"))

    def test_unknown_extension_import_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot determine format"):
            import_model(str(tmp_path / "model.xyz"))

    def test_constraints_preserved(self, tmp_path):
        """Constraints should survive the roundtrip."""
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y >= 2)  # forces both to 1
        model.minimize(x + y)

        filepath = str(tmp_path / "model.pb")
        export_model(model, filepath)
        loaded = import_model(filepath)
        assert_objective(loaded, expected=2.0)

    def test_empty_model_roundtrip(self, tmp_path):
        """An empty model should roundtrip without errors."""
        model = cp_model.CpModel()
        for ext in (".pb", ".pbtxt"):
            filepath = str(tmp_path / f"empty{ext}")
            export_model(model, filepath)
            loaded = import_model(filepath)
            assert loaded is not None
