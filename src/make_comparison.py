"""Create before/after montages for manual inspection.

This script does not need a trained model or a YOLO dataset.  It is useful
while waiting for the dataset/model: pick a few images, run one or two
methods, and keep the comparison images for the report.

Usage::

    python -m src.make_comparison --input path/to/images --methods clahe median

The montage is resized for viewing only.  The images used by the evaluation
script keep their original size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .preprocessing import METHODS, preprocess_image
except ImportError:  # running as a plain script
    from preprocessing import METHODS, preprocess_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
HEADER_HEIGHT = 36
GAP = 12
SELECTABLE_METHODS = tuple(method for method in METHODS if method != "none")


def _load_font(size: int = 20) -> ImageFont.ImageFont:
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
    title: str,
    font: ImageFont.ImageFont,
) -> Image.Image:
    width = before.width + GAP + after.width
    height = HEADER_HEIGHT + max(before.height, after.height)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    canvas.paste(before, (0, HEADER_HEIGHT))
    canvas.paste(after, (before.width + GAP, HEADER_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, HEADER_HEIGHT), fill=(30, 30, 30))
    draw.text((8, 8), "Original", font=font, fill=(255, 255, 255))
    draw.text(
        (before.width + GAP + 8, 8),
        title,
        font=font,
        fill=(255, 255, 255),
    )
    return canvas


def _collect_images(input_path: Path) -> tuple[Path, list[Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(f"not a supported image: {input_path}")
        return input_path.parent, [input_path]

    if not input_path.is_dir():
        raise SystemExit(f"input not found: {input_path}")

    images = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"no images found in: {input_path}")
    return input_path, images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create original-vs-preprocessed montages."
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
        default=Path("results/preprocessing"),
        help="output directory (default: results/preprocessing)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["clahe"],
        choices=SELECTABLE_METHODS,
        help="one or more methods to compare",
    )
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
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="also save the full-size processed image",
    )
    args = parser.parse_args()

    root, images = _collect_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]

    font = _load_font()
    output_root: Path = args.output
    saved = 0
    failures: list[tuple[Path, Exception]] = []

    for image_path in images:
        relative = image_path.relative_to(root)
        target_dir = output_root / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(image_path) as opened:
                original = opened.convert("RGB")
            array = np.asarray(original)

            for method in args.methods:
                processed = preprocess_image(array, method)
                after = Image.fromarray(processed)
                safe_method = method.replace("+", "_")

                if args.save_individual:
                    after.save(
                        target_dir / f"{relative.stem}_{safe_method}.jpg",
                        quality=95,
                    )

                montage = _montage(
                    _fit(original, args.max_panel_width),
                    _fit(after, args.max_panel_width),
                    method.upper(),
                    font,
                )
                destination = (
                    target_dir
                    / f"{relative.stem}_{safe_method}_compare.jpg"
                )
                montage.save(destination, quality=92)
                saved += 1
        except Exception as exc:  # keep going if one file is broken
            failures.append((image_path, exc))
            print(f"[skip] {image_path}: {exc}")

    print(f"\nSaved {saved} comparison image(s) to {output_root}")
    if failures:
        print(f"Failed: {len(failures)}")
        for path, exc in failures:
            print(f"  - {path}: {exc}")


if __name__ == "__main__":
    main()
