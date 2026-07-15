import os
from pathlib import Path

import pytest

from app.ml.common import paths


def test_repository_root_detection():
    root = paths.get_repository_root()

    assert root.is_absolute()
    assert (root / "backend").is_dir()
    assert (root / "Final Dataset").is_dir()


def test_backend_root_detection():
    backend_root = paths.get_backend_root()

    assert backend_root.is_absolute()
    assert backend_root.name == "backend"
    assert (backend_root / "app").is_dir()


def test_raw_dataset_root_resolution():
    raw_root = paths.get_raw_dataset_root()

    assert raw_root == paths.get_repository_root() / "Final Dataset"
    assert raw_root.is_absolute()


def test_canonical_model_root_resolution():
    model_root = paths.get_model_root()

    assert model_root == paths.get_repository_root() / "ml_models"
    assert model_root.is_absolute()


@pytest.mark.parametrize("value", ["../ml_models", "./ml_models", "ml_models", ".\\ml_models", "..\\ml_models"])
def test_model_root_relative_path_resolution(value):
    resolved = paths.resolve_model_root(value)

    assert resolved == paths.get_repository_root() / "ml_models"


def test_generated_directory_creation():
    created = paths.ensure_generated_directories()

    assert paths.get_generated_root() in created
    for directory in created:
        assert directory.is_dir()
        assert not paths.is_path_inside(paths.get_raw_dataset_root(), directory)
        assert any(
            paths.is_path_inside(root, directory)
            for root in (
                paths.get_generated_root(),
                paths.get_ml_research_root(),
                paths.get_repository_root() / "ml_models",
            )
        )


def test_model_directory_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "settings", type("SettingsStub", (), {"MODEL_ROOT": str(tmp_path / "models")})())

    model_dir = paths.ensure_model_directory("phase2-test-artifacts")

    assert model_dir.is_dir()
    assert paths.is_path_inside(paths.get_model_root(), model_dir)
    assert not paths.is_path_inside(paths.get_raw_dataset_root(), model_dir)


def test_rejects_writes_inside_raw_dataset():
    with pytest.raises(ValueError, match="raw dataset"):
        paths.assert_not_raw_dataset_path(paths.get_raw_dataset_root() / "new_file.csv")


def test_rejects_generated_path_traversal():
    with pytest.raises(ValueError, match="traversal"):
        paths.resolve_generated_path("../Final Dataset/unsafe.csv")


def test_generated_outputs_remain_outside_raw_dataset_paths():
    output = paths.resolve_generated_path("preprocessing/text/features.parquet")

    assert paths.is_path_inside(paths.get_generated_root(), output)
    assert not paths.is_path_inside(paths.get_raw_dataset_root(), output)


def test_paths_do_not_depend_on_current_working_directory(tmp_path, monkeypatch):
    before = {
        "repo": paths.get_repository_root(),
        "backend": paths.get_backend_root(),
        "raw": paths.get_raw_dataset_root(),
        "model": paths.get_model_root(),
    }

    monkeypatch.chdir(tmp_path)

    after = {
        "repo": paths.get_repository_root(),
        "backend": paths.get_backend_root(),
        "raw": paths.get_raw_dataset_root(),
        "model": paths.get_model_root(),
    }
    assert after == before


def test_new_path_module_has_no_machine_specific_absolute_paths():
    source = Path(paths.__file__).read_text(encoding="utf-8")

    assert "D:\\" not in source
    assert "C:\\" not in source


def test_custom_relative_model_root_resolves_against_backend_root():
    resolved = paths.resolve_model_root("custom_models")

    assert resolved == paths.get_backend_root() / "custom_models"


def test_absolute_model_root_override_is_preserved(tmp_path):
    resolved = paths.resolve_model_root(tmp_path / "models")

    assert resolved == (tmp_path / "models").resolve()


def test_generated_path_rejects_absolute_paths(tmp_path):
    with pytest.raises(ValueError, match="relative path"):
        paths.resolve_generated_path(tmp_path / "features.csv")


def test_model_directory_rejects_traversal():
    with pytest.raises(ValueError, match="traversal"):
        paths.ensure_model_directory("../unsafe")
