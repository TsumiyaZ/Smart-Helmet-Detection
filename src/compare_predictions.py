"""Compare model detections on original vs preprocessed images.

This is a *smoke test*, not a replacement for precision/recall/mAP.  It shows
whether the same model finds different objects after preprocessing and saves
annotated before/after images for the report.

Usage::

    python -m src.compare_predictions --input images --methods clahe median

For real metrics use ``src/evaluate_preprocessing.py`` with a labelled
validation/test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .detection import predict_image
    from .preprocessing import METHODS, preprocess_image
except ImportError:  # running as a plain script
    from detection import predict_image
    from preprocessing import METHODS, preprocess_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
HEADER_HEIGHT = 44
GAP = 12
SELECTABLE_METHODS = tuple(method for method in METHODS if method != "none")

CSV_FIELDS = [
    "image",
    "method",
    "confidence",
    "original_total",
    "processed_total",
    "original_counts",
    "processed_counts",
    "original_detections",
    "processed_detections",
]


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fit(image: Image.Image, max_width: int) -> Image.Image:
    if max_width <= 0 or image.width <= max_width:
        return image
    ratio = max_width / image.width
    return image.resize(
        (max_width, max(1, round(image.height * ratio))),
        Image.Resampling.LANCZOS,
    )


def _montage(
    before: Image.Image,
    after: Image.Image,
    before_label: str,
    after_label: str,
    font: ImageFont.ImageFont,
) -> Image.Image:
    width = before.width + GAP + after.width
    height = HEADER_HEIGHT + max(before.height, after.height)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    canvas.paste(before, (0, HEADER_HEIGHT))
    canvas.paste(after, (before.width + GAP, HEADER_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, HEADER_HEIGHT), fill=(30, 30, 30))
    draw.text((8, 12), before_label, font=font, fill=(255, 255, 255))
    draw.text(
        (before.width + GAP + 8, 12),
        after_label,
        font=font,
        fill=(255, 255, 255),
    )
    return canvas


def _counts_text(counts: dict[str, int]) -> str:
    if not counts:
        return "0"
    return ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(f"not a supported image: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"input not found: {input_path}")
    images = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"no images found in: {input_path}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare detections before/after preprocessing (smoke test)."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="image file or directory (searched recursively)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/prediction_comparison"),
        help="output directory",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["clahe"],
        choices=SELECTABLE_METHODS,
        help="one or more preprocessing methods",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only the first N images; 0 means all",
    )
    parser.add_argument(
        "--max-panel-width",
        type=int,
        default=960,
        help="resize each panel for the montage; 0 keeps full size",
    )
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")

    images = _collect_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    font = _load_font()
    rows: list[dict[str, Any]] = []
    failures: list[tuple[Path, Exception]] = []

    for image_path in images:
        try:
            with Image.open(image_path) as opened:
                image_rgb = np.asarray(opened.convert("RGB"))

            original = predict_image(image_rgb, args.confidence)
            original_total = sum(original["counts"].values())

            for method in args.methods:
                processed = preprocess_image(image_rgb, method)
                result = predict_image(processed, args.confidence)
                processed_total = sum(result["counts"].values())

                before = Image.fromarray(original["annotated_rgb"])
                after = Image.fromarray(result["annotated_rgb"])
                montage = _montage(
                    _fit(before, args.max_panel_width),
                    _fit(after, args.max_panel_width),
                    f"Original ({original_total})",
                    f"{method.upper()} ({processed_total})",
                    font,
                )
                destination = (
                    args.output
                    / f"{image_path.stem}_{method.replace('+', '_')}_detections.jpg"
                )
                montage.save(destination, quality=92)

                rows.append(
                    {
                        "image": str(image_path),
                        "method": method,
                        "confidence": args.confidence,
                        "original_total": original_total,
                        "processed_total": processed_total,
                        "original_counts": json.dumps(
                            original["counts"], ensure_ascii=False
                        ),
                        "processed_counts": json.dumps(
                            result["counts"], ensure_ascii=False
                        ),
                        "original_detections": json.dumps(
                            original["detections"], ensure_ascii=False
                        ),
                        "processed_detections": json.dumps(
                            result["detections"], ensure_ascii=False
                        ),
                    }
                )
                print(
                    f"{image_path.name} [{method}] "
                    f"original={_counts_text(original['counts'])} | "
                    f"processed={_counts_text(result['counts'])}"
                )
        except Exception as exc:  # keep going if one image is broken
            failures.append((image_path, exc))
            print(f"[skip] {image_path}: {exc}")

    csv_path = args.output / "prediction_smoke_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} comparison(s) to {args.output}")
    print(f"CSV: {csv_path}")
    if failures:
        print(f"Failed: {len(failures)}")
        for path, exc in failures:
            print(f"  - {path}: {exc}")


if __name__ == "__main__":
    main()
