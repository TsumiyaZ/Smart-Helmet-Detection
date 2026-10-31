"""Compare YOLO metrics on the original vs preprocessed validation set.

The comparison keeps every condition identical except preprocessing:

* the same model file (its SHA-256 is stored in the result);
* the same image list (its SHA-256 is stored in the result);
* the same split, confidence threshold and image size.

Usage::

    # Prepare the preprocessed split now; a model is not required.
    python -m src.evaluate_preprocessing --data data.yaml --method clahe --prepare-only

    # Run the real comparison after best.pt and the dataset are ready.
    python -m src.evaluate_preprocessing --data data.yaml --model best.pt --method clahe

The script materialises the preprocessed images in a generated folder so that
``model.val()`` can read them like a normal YOLO dataset.  The folder is
removed after a successful comparison unless ``--keep-prepared`` is given.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

try:
    from .preprocessing import METHODS, preprocess_image
except ImportError:  # running as a plain script
    from preprocessing import METHODS, preprocess_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SELECTABLE_METHODS = tuple(method for method in METHODS if method != "none")
MARKER = ".generated-by-evaluate-preprocessing"

CSV_FIELDS = [
    "timestamp",
    "experiment",
    "model",
    "model_sha256",
    "data_yaml",
    "split",
    "image_count",
    "image_list_sha256",
    "method",
    "params",
    "conf",
    "imgsz",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "notes",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"data.yaml is not a mapping: {path}")
    return data


def _resolve_dir(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _normalise_split(split: str) -> str:
    aliases = {"validation": "val", "training": "train"}
    return aliases.get(split.strip().lower(), split.strip().lower())


def _resolve_split(
    data: dict[str, Any],
    yaml_path: Path,
    split: str,
) -> tuple[str, Path]:
    split = _normalise_split(split)
    if split not in data:
        raise SystemExit(f"split {split!r} not found in {yaml_path}")

    value = data[split]
    if not isinstance(value, str):
        raise SystemExit(
            "only a single directory per split is supported; "
            f"got {type(value).__name__} for {split!r}"
        )

    root_value = data.get("path")
    root = (
        yaml_path.parent
        if root_value is None
        else _resolve_dir(str(root_value), yaml_path.parent)
    )
    images_dir = _resolve_dir(value, root)
    if not images_dir.is_dir():
        raise SystemExit(f"image directory not found: {images_dir}")
    return split, images_dir


def _find_labels_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            candidate = Path(*parts[:index], "labels", *parts[index + 1 :])
            if candidate.is_dir():
                return candidate

    candidates = [
        images_dir.parent / "labels",
        images_dir.parent / f"{images_dir.name}_labels",
        images_dir.parent.parent / "labels" / images_dir.name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise SystemExit(
        "could not find the label directory for "
        f"{images_dir}; expected a sibling 'labels' directory"
    )


def _collect_images(images_dir: Path, limit: int = 0) -> list[Path]:
    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"no images found in: {images_dir}")
    if limit > 0:
        images = images[:limit]
    return images


def _image_list_hash(images: list[Path], images_dir: Path) -> str:
    lines = sorted(
        str(path.relative_to(images_dir)).replace("\\", "/")
        for path in images
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_split(
    data_yaml: Path,
    split: str,
    method: str,
    params: dict[str, Any],
    prepared_dir: Path,
    limit: int = 0,
) -> dict[str, Any]:
    data = _load_yaml(data_yaml)
    split, images_dir = _resolve_split(data, data_yaml, split)
    labels_dir = _find_labels_dir(images_dir)
    images = _collect_images(images_dir, limit)

    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise SystemExit(
            "duplicate image stems would collide in the generated dataset; "
            "rename the files before evaluating"
        )

    if prepared_dir.exists():
        marker = prepared_dir / MARKER
        if prepared_dir.is_dir() and (
            marker.exists() or not any(prepared_dir.iterdir())
        ):
            shutil.rmtree(prepared_dir)
        else:
            raise SystemExit(
                f"refusing to overwrite {prepared_dir}; "
                "remove it manually or choose another --prepared-dir"
            )

    image_out_dir = prepared_dir / "images" / split
    label_out_dir = prepared_dir / "labels" / split
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / MARKER).write_text(
        "Generated by src/evaluate_preprocessing.py\n",
        encoding="utf-8",
    )

    missing_labels = 0
    for image_path in images:
        with Image.open(image_path) as opened:
            image_rgb = np.asarray(opened.convert("RGB"))

        processed = preprocess_image(image_rgb, method, **params)
        Image.fromarray(processed).save(
            image_out_dir / f"{image_path.stem}.jpg",
            quality=95,
        )

        source_label = labels_dir / f"{image_path.stem}.txt"
        target_label = label_out_dir / f"{image_path.stem}.txt"
        if source_label.is_file():
            shutil.copyfile(source_label, target_label)
        else:
            target_label.write_text("", encoding="utf-8")
            missing_labels += 1

    names = data.get("names")
    if names is None:
        raise SystemExit(f"'names' is missing from {data_yaml}")

    prepared_data: dict[str, Any] = {
        "path": prepared_dir.resolve().as_posix(),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "names": names,
    }
    if "nc" in data:
        prepared_data["nc"] = data["nc"]

    prepared_yaml = prepared_dir / "data.yaml"
    with prepared_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            prepared_data,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )

    return {
        "split": split,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "images": images,
        "prepared_dir": prepared_dir,
        "prepared_yaml": prepared_yaml,
        "missing_labels": missing_labels,
    }


def _run_val(
    model_path: Path,
    data_yaml: Path,
    split: str,
    output_dir: Path,
    imgsz: int,
    conf: float,
    device: str | None,
    workers: int,
    plots: bool,
    run_name: str,
) -> dict[str, float]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on local setup
        raise SystemExit(
            "ultralytics is not installed. Install it with: "
            "pip install -r requirements.txt"
        ) from exc

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        device=device,
        workers=workers,
        project=str(output_dir / "runs"),
        name=run_name,
        exist_ok=True,
        plots=plots,
        verbose=False,
    )

    results = metrics.results_dict or {}
    box = metrics.box
    # dict.get(key, default) always evaluates its default, so look up the
    # Ultralytics attributes lazily instead.
    def pick(key: str, attribute: str) -> float:
        value = results.get(key)
        if value is None:
            value = getattr(box, attribute, None)
        if value is None:
            raise RuntimeError(f"metric not found: {key} (or box.{attribute})")
        return float(value)

    return {
        "precision": pick("metrics/precision(B)", "mp"),
        "recall": pick("metrics/recall(B)", "mr"),
        "map50": pick("metrics/mAP50(B)", "map50"),
        "map50_95": pick("metrics/mAP50-95(B)", "map"),
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _write_csv(
    csv_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    has_header = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not has_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    metadata: dict[str, Any],
    baseline: dict[str, float],
    processed: dict[str, float],
) -> None:
    delta = {
        key: processed[key] - baseline[key]
        for key in baseline
    }

    lines = [
        "# เปรียบเทียบผลก่อน/หลังปรับภาพ",
        "",
        f"- วันที่: {metadata['timestamp']}",
        (
            f"- โมเดล: `{metadata['model_name']}` "
            f"(SHA-256 `{metadata['model_sha256'][:12]}…`)"
        ),
        (
            f"- ชุดภาพ: `{metadata['data_yaml']}` / split "
            f"`{metadata['split']}` / {metadata['image_count']} ภาพ "
            f"(SHA-256 `{metadata['image_list_sha256'][:12]}…`)"
        ),
        (
            f"- วิธีปรับภาพ: `{metadata['method']}` "
            f"params `{metadata['params']}`"
        ),
        (
            f"- เงื่อนไข: confidence `{metadata['conf']}`, "
            f"imgsz `{metadata['imgsz']}` (ค่าเดียวกันทั้งสองรอบ)"
        ),
        "",
        "| การทดลอง | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| ภาพเดิม | {_fmt(baseline['precision'])} | "
            f"{_fmt(baseline['recall'])} | {_fmt(baseline['map50'])} | "
            f"{_fmt(baseline['map50_95'])} |"
        ),
        (
            f"| หลังปรับ (`{metadata['method']}`) | "
            f"{_fmt(processed['precision'])} | "
            f"{_fmt(processed['recall'])} | "
            f"{_fmt(processed['map50'])} | "
            f"{_fmt(processed['map50_95'])} |"
        ),
        (
            f"| ผลต่าง (Δ) | {delta['precision']:+.4f} | "
            f"{delta['recall']:+.4f} | {delta['map50']:+.4f} | "
            f"{delta['map50_95']:+.4f} |"
        ),
        "",
        "> ตัวเลขทั้งหมดมาจาก `ultralytics.YOLO.val()` โดยตรง "
        "ไม่ได้คำนวณหรือปรับค่าเอง",
    ]
    if metadata["notes"]:
        lines.extend(["", f"หมายเหตุ: {metadata['notes']}"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare YOLO validation metrics before and after preprocessing."
        )
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="original data.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="trained best.pt (required unless --prepare-only)",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="dataset split to evaluate (default: val)",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=SELECTABLE_METHODS,
        help="preprocessing method to evaluate",
    )
    parser.add_argument(
        "--params",
        default="{}",
        help='JSON parameters, e.g. \'{"clip_limit": 3.0}\'',
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, e.g. 0 or cpu (default: auto)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="dataloader workers (default 0 is safest on Windows)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation"),
        help="result directory (default: results/evaluation)",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=None,
        help="where to write the preprocessed split",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="only build the preprocessed split; no model needed",
    )
    parser.add_argument(
        "--keep-prepared",
        action="store_true",
        help="keep the generated preprocessed split after comparing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only the first N images; 0 means all (subset warning added)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="let Ultralytics save PR curves and other plots",
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be a positive integer")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1")

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--params is not valid JSON: {exc}") from exc
    if not isinstance(params, dict):
        raise SystemExit("--params must be a JSON object")

    split = _normalise_split(args.split)
    safe_method = args.method.replace("+", "_")
    prepared_dir = args.prepared_dir or (
        args.output / "prepared" / f"{safe_method}_{split}"
    )

    model_path: Path | None = None
    if not args.prepare_only:
        if args.model is None:
            raise SystemExit(
                "--model is required unless --prepare-only is used"
            )
        model_path = args.model.resolve()
        if not model_path.is_file():
            raise SystemExit(f"model not found: {model_path}")
        if model_path.stat().st_size == 0:
            raise SystemExit(
                f"model file is empty (0 bytes): {model_path}; "
                "copy the trained best.pt from Colab/Drive first"
            )

    info = _prepare_split(
        args.data.resolve(),
        split,
        args.method,
        params,
        prepared_dir.resolve(),
        args.limit,
    )

    notes = args.notes.strip()
    if args.limit > 0:
        limit_note = f"ใช้ภาพเพียง {args.limit} ภาพแรก (subset)"
        notes = f"{notes}; {limit_note}" if notes else limit_note
    if info["missing_labels"]:
        missing_note = (
            f"ไม่พบป้ายกำกับ {info['missing_labels']} ภาพ "
            "จึงใช้ไฟล์ป้ายว่าง"
        )
        notes = f"{notes}; {missing_note}" if notes else missing_note

    print(
        f"Prepared {len(info['images'])} images "
        f"({args.method}) at {info['prepared_dir']}"
    )
    if info["missing_labels"]:
        print(
            f"Warning: {info['missing_labels']} image(s) had no label file; "
            "empty labels were written."
        )

    if args.prepare_only:
        print(f"data.yaml: {info['prepared_yaml']}")
        print("Prepared only; no model was run.")
        return

    assert model_path is not None  # checked above when not --prepare-only

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating original images with {model_path.name} ...")
    baseline = _run_val(
        model_path=model_path,
        data_yaml=args.data.resolve(),
        split=split,
        output_dir=output_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        workers=args.workers,
        plots=args.plots,
        run_name=f"original_{timestamp}",
    )

    print(f"Evaluating preprocessed images ({args.method}) ...")
    processed = _run_val(
        model_path=model_path,
        data_yaml=info["prepared_yaml"],
        split=split,
        output_dir=output_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        workers=args.workers,
        plots=args.plots,
        run_name=f"preprocessed_{safe_method}_{timestamp}",
    )

    metadata = {
        "timestamp": timestamp,
        "model_name": model_path.name,
        "model_sha256": _sha256_file(model_path),
        "data_yaml": str(args.data.resolve()),
        "split": info["split"],
        "image_count": len(info["images"]),
        "image_list_sha256": _image_list_hash(
            info["images"], info["images_dir"]
        ),
        "method": args.method,
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "notes": notes,
    }

    common = {
        "timestamp": timestamp,
        "model": str(model_path),
        "model_sha256": metadata["model_sha256"],
        "data_yaml": metadata["data_yaml"],
        "split": info["split"],
        "image_count": len(info["images"]),
        "image_list_sha256": metadata["image_list_sha256"],
        "params": metadata["params"],
        "conf": args.conf,
        "imgsz": args.imgsz,
        "notes": notes,
    }
    rows = [
        {
            **common,
            "experiment": "original",
            "method": "none",
            **{key: baseline[key] for key in baseline},
        },
        {
            **common,
            "experiment": "preprocessed",
            "method": args.method,
            **{key: processed[key] for key in processed},
        },
    ]

    csv_path = output_dir / "preprocessing_comparison.csv"
    _write_csv(csv_path, rows)

    markdown_path = output_dir / "preprocessing_comparison.md"
    _write_markdown(markdown_path, metadata, baseline, processed)
    markdown_path.with_name(
        f"preprocessing_comparison_{timestamp}.md"
    ).write_text(
        markdown_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print("\nResults")
    print(f"  original     : {baseline}")
    print(f"  {args.method:<12}: {processed}")
    print(f"  CSV          : {csv_path}")
    print(f"  Markdown     : {markdown_path}")

    if not args.keep_prepared:
        shutil.rmtree(info["prepared_dir"], ignore_errors=True)
        print(f"Removed prepared split: {info['prepared_dir']}")
    else:
        print(f"Kept prepared split: {info['prepared_dir']}")


if __name__ == "__main__":
    main()
