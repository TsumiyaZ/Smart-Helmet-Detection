"""Tests for the dataset preparation and result files in
``src/evaluate_preprocessing.py``.

These tests do not run YOLO; they only verify the parts that must work while
waiting for a labelled dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.evaluate_preprocessing import (
    _image_list_hash,
    _prepare_split,
    _write_csv,
    _write_markdown,
)


def _make_dataset(root: Path) -> Path:
    images_dir = root / "dataset" / "images" / "val"
    labels_dir = root / "dataset" / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    rng = np.random.default_rng(1)
    for name in ("a", "b"):
        Image.fromarray(
            rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
        ).save(images_dir / f"{name}.jpg")
        (labels_dir / f"{name}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    data = {
        "path": (root / "dataset").as_posix(),
        "train": "images/val",
        "val": "images/val",
        "test": "images/val",
        "names": {0: "helmet", 1: "no_helmet"},
    }
    yaml_path = root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return yaml_path


def test_prepare_split_creates_yolo_layout(tmp_path: Path) -> None:
    yaml_path = _make_dataset(tmp_path)
    prepared = tmp_path / "prepared"

    info = _prepare_split(yaml_path, "val", "clahe", {}, prepared)

    assert info["split"] == "val"
    assert len(info["images"]) == 2
    assert info["missing_labels"] == 0
    assert (prepared / "images" / "val" / "a.jpg").is_file()
    assert (prepared / "labels" / "val" / "a.txt").read_text(
        encoding="utf-8"
    ).startswith("0 ")

    generated = yaml.safe_load(
        (prepared / "data.yaml").read_text(encoding="utf-8")
    )
    assert generated["names"] == {0: "helmet", 1: "no_helmet"}
    assert generated["val"] == "images/val"

    with Image.open(prepared / "images" / "val" / "a.jpg") as image:
        assert image.size == (48, 32)


def test_prepare_split_can_limit_and_writes_empty_label(
    tmp_path: Path,
) -> None:
    yaml_path = _make_dataset(tmp_path)
    (tmp_path / "dataset" / "labels" / "val" / "a.txt").unlink()

    info = _prepare_split(
        yaml_path,
        "val",
        "median",
        {},
        tmp_path / "prepared",
        limit=1,
    )

    assert len(info["images"]) == 1
    assert info["missing_labels"] == 1
    assert (
        tmp_path / "prepared" / "labels" / "val" / "a.txt"
    ).read_text(encoding="utf-8") == ""


def test_image_list_hash_is_stable(tmp_path: Path) -> None:
    yaml_path = _make_dataset(tmp_path)
    info = _prepare_split(yaml_path, "val", "none", {}, tmp_path / "prepared")

    first = _image_list_hash(info["images"], info["images_dir"])
    second = _image_list_hash(
        list(reversed(info["images"])),
        info["images_dir"],
    )
    assert first == second


def test_csv_header_is_written_once(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    rows = [
        {
            "timestamp": "20260101_000000",
            "experiment": "original",
            "model": "best.pt",
            "model_sha256": "abc",
            "data_yaml": "data.yaml",
            "split": "val",
            "image_count": 2,
            "image_list_sha256": "def",
            "method": "none",
            "params": "{}",
            "conf": 0.25,
            "imgsz": 640,
            "precision": 0.5,
            "recall": 0.5,
            "map50": 0.5,
            "map50_95": 0.4,
            "notes": "",
        }
    ]

    _write_csv(csv_path, rows)
    _write_csv(csv_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))
    assert len(parsed) == 2
    assert parsed[0]["experiment"] == "original"


def test_markdown_contains_both_experiments(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"
    metadata = {
        "timestamp": "20260101_000000",
        "model_name": "best.pt",
        "model_sha256": "a" * 64,
        "data_yaml": "data.yaml",
        "split": "val",
        "image_count": 2,
        "image_list_sha256": "b" * 64,
        "method": "clahe",
        "params": "{}",
        "conf": 0.25,
        "imgsz": 640,
        "notes": "",
    }
    baseline = {
        "precision": 0.5,
        "recall": 0.4,
        "map50": 0.45,
        "map50_95": 0.3,
    }
    processed = {
        "precision": 0.6,
        "recall": 0.5,
        "map50": 0.55,
        "map50_95": 0.4,
    }

    _write_markdown(path, metadata, baseline, processed)

    text = path.read_text(encoding="utf-8")
    assert "ภาพเดิม" in text
    assert "clahe" in text
    assert "0.5500" in text
    assert "+0.1000" in text
