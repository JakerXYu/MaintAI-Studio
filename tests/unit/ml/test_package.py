"""Package tests: roundtrip, predict, hash, corrupt, traversal, extension, atomic."""

import hashlib
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from maintai.ml import package
from maintai.ml.preprocess import build_preprocessor


def _fit_pipeline():
    rng = np.random.default_rng(0)
    n = 80
    frame = pd.DataFrame(
        {
            "num": rng.normal(0, 1, n),
            "cat": rng.choice(["a", "b"], n),
            "target": (rng.normal(0, 1, n) > 0).astype(int),
        }
    )
    features = ["num", "cat"]
    X = frame[features]
    preprocessor = build_preprocessor(X, features)
    preprocessor.fit(X)
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]
    )
    pipeline.fit(X, frame["target"])
    return pipeline, X


def _save(pipeline, tmp_path):
    return package.save_artifact(
        pipeline,
        model_name="rf_test",
        version="1.2.3",
        task="binary_classification",
        features=["num", "cat"],
        input_schema={"num": "float", "cat": "str"},
        training_summary={"f1": 0.9},
        artifact_dir=tmp_path,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _read_members(path):
    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        pipeline_bytes = archive.read("pipeline.joblib")
    return manifest, pipeline_bytes


def _write_members(path, manifest, pipeline_bytes):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))
        archive.writestr("pipeline.joblib", pipeline_bytes)


def test_roundtrip_and_predict(tmp_path):
    pipeline, X = _fit_pipeline()
    path = _save(pipeline, tmp_path)
    name = os.path.basename(path)
    assert name.endswith(".joblib")

    loaded, manifest = package.load_artifact(tmp_path, name)
    assert manifest.model_name == "rf_test"
    assert manifest.version == "1.2.3"
    assert manifest.task == "binary_classification"
    assert manifest.features == ["num", "cat"]
    assert manifest.input_schema == {"num": "float", "cat": "str"}
    assert manifest.training_summary == {"f1": 0.9}
    assert manifest.created_at == "2026-01-01T00:00:00+00:00"
    assert manifest.manifest_version == package.MANIFEST_VERSION
    assert len(manifest.sha256) == 64

    np.testing.assert_array_equal(loaded.predict(X), pipeline.predict(X))


def test_hash_matches_stored_bytes(tmp_path):
    pipeline, X = _fit_pipeline()
    path = _save(pipeline, tmp_path)
    name = os.path.basename(path)
    _, manifest = package.load_artifact(tmp_path, name)

    _, pipeline_bytes = _read_members(path)
    digest = hashlib.sha256(pipeline_bytes).hexdigest()
    assert manifest.sha256 == digest


def test_tampered_sha256_rejected(tmp_path):
    pipeline, X = _fit_pipeline()
    path = _save(pipeline, tmp_path)
    manifest, pipeline_bytes = _read_members(path)
    manifest["sha256"] = "0" * 64
    _write_members(tmp_path / "tampered.joblib", manifest, pipeline_bytes)
    with pytest.raises(package.ArtifactError, match="SHA-256 mismatch"):
        package.load_artifact(tmp_path, "tampered.joblib")


def test_unsupported_manifest_version_rejected(tmp_path):
    pipeline, X = _fit_pipeline()
    path = _save(pipeline, tmp_path)
    manifest, pipeline_bytes = _read_members(path)
    manifest["manifest_version"] = 999
    _write_members(tmp_path / "old.joblib", manifest, pipeline_bytes)
    with pytest.raises(package.ArtifactError, match="unsupported manifest version"):
        package.load_artifact(tmp_path, "old.joblib")


def test_corrupt_file_rejected(tmp_path):
    (tmp_path / "corrupt.joblib").write_bytes(b"not a joblib file at all")
    with pytest.raises(package.ArtifactError):
        package.load_artifact(tmp_path, "corrupt.joblib")


def test_path_traversal_rejected(tmp_path):
    for name in ("../evil.joblib", "..\\evil.joblib", "sub/evil.joblib", "sub\\evil.joblib"):
        with pytest.raises(package.ArtifactError):
            package.load_artifact(tmp_path, name)


def test_untrusted_extension_rejected(tmp_path):
    for name in ("model.pkl", "model.txt", "model"):
        with pytest.raises(package.ArtifactError):
            package.load_artifact(tmp_path, name)


def test_missing_artifact_rejected(tmp_path):
    with pytest.raises(package.ArtifactError, match="not found"):
        package.load_artifact(tmp_path, "missing.joblib")


def test_atomic_no_leftover_tmp(tmp_path):
    pipeline, X = _fit_pipeline()
    _save(pipeline, tmp_path)
    leftovers = [entry for entry in tmp_path.iterdir() if entry.suffix == ".tmp"]
    assert leftovers == []


def test_save_to_non_directory_rejected(tmp_path):
    pipeline, X = _fit_pipeline()
    file_path = tmp_path / "afile"
    file_path.write_text("x")
    with pytest.raises(package.ArtifactError):
        package.save_artifact(
            pipeline,
            model_name="m",
            version="1",
            task="t",
            features=["num"],
            artifact_dir=file_path,
        )
