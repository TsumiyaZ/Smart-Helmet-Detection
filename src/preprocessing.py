"""RGB image preprocessing for the Smart Helmet Detection project.

Shared contract (see ``Obsidian-work/03 - สัญญาเชื่อมระบบ.md``):

* input and output are RGB ``uint8`` arrays with shape ``(H, W, 3)``;
* the output keeps the original width, height and dtype;
* the input array is never modified in place.

The two techniques required by the assignment are CLAHE and a median filter.
They can be evaluated separately or together.  The method is intentionally
selectable so the team can turn it on/off from the web app.

Command line example::

    python -m src.preprocessing input.jpg output.jpg --method clahe
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on local setup
    raise ImportError(
        "OpenCV is required for image preprocessing. "
        "Install it with: pip install -r requirements.txt"
    ) from exc


METHODS = ("none", "clahe", "median", "clahe+median")
DEFAULT_METHOD = "none"

_CLAHE_DEFAULTS = {"clip_limit": 2.0, "tile_grid_size": (8, 8)}
_MEDIAN_DEFAULTS = {"median_ksize": 3}


def available_methods() -> tuple[str, ...]:
    """Return all method names accepted by :func:`preprocess_image`."""
    return METHODS


def _validate_rgb(image_rgb: Any) -> None:
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError(
            f"image_rgb must be a numpy.ndarray, got {type(image_rgb).__name__}"
        )
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            "image_rgb must have shape (H, W, 3); "
            f"got {getattr(image_rgb, 'shape', None)!r}"
        )
    if image_rgb.size == 0:
        raise ValueError("image_rgb is empty")
    if image_rgb.dtype != np.uint8:
        raise TypeError(
            "image_rgb must be uint8 in the range 0..255; "
            f"got dtype {image_rgb.dtype}"
        )


def _take_params(
    params: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    unknown = set(params) - set(defaults)
    if unknown:
        allowed = ", ".join(sorted(defaults))
        raise TypeError(
            f"unknown parameter(s): {', '.join(sorted(unknown))}; "
            f"allowed: {allowed}"
        )
    merged = dict(defaults)
    merged.update(params)
    return merged


def _clahe(
    image_rgb: np.ndarray,
    *,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    clip_limit = float(clip_limit)
    if clip_limit <= 1.0:
        raise ValueError("clip_limit must be greater than 1.0")

    try:
        tile = tuple(int(value) for value in tile_grid_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tile_grid_size must contain two positive integers"
        ) from exc
    if len(tile) != 2 or min(tile) < 1:
        raise ValueError("tile_grid_size must contain two positive integers")

    # Apply CLAHE only to lightness so the colours stay close to the original.
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2RGB,
    )


def _median(image_rgb: np.ndarray, *, median_ksize: int) -> np.ndarray:
    ksize = int(median_ksize)
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(
            "median_ksize must be an odd integer greater than or equal to 3"
        )
    return cv2.medianBlur(image_rgb, ksize)


def preprocess_image(
    image_rgb: np.ndarray,
    method: str = DEFAULT_METHOD,
    **params: Any,
) -> np.ndarray:
    """Return a preprocessed copy of an RGB image.

    Parameters
    ----------
    image_rgb:
        RGB ``uint8`` array with shape ``(H, W, 3)``.
    method:
        ``"none"``, ``"clahe"``, ``"median"`` or ``"clahe+median"``.
    **params:
        * ``clip_limit`` (float > 1.0, default 2.0) — CLAHE strength.
        * ``tile_grid_size`` (two positive ints, default ``(8, 8)``).
        * ``median_ksize`` (odd int >= 3, default 3) — median kernel.

    Raises
    ------
    TypeError, ValueError
        If the image, method or parameters are invalid.

    Notes
    -----
    CLAHE is applied to the L channel in CIELAB, so it brightens local
    contrast without swapping the red/green/blue channels.  The median
    filter works directly on the three RGB channels.
    """
    _validate_rgb(image_rgb)
    if method is not None and not isinstance(method, str):
        raise TypeError(
            f"method must be a string, got {type(method).__name__}"
        )
    method = (method or DEFAULT_METHOD).strip().lower()

    if method == "none":
        _take_params(params, {})
        return image_rgb.copy()

    if method == "clahe":
        values = _take_params(params, _CLAHE_DEFAULTS)
        output = _clahe(image_rgb, **values)
    elif method == "median":
        values = _take_params(params, _MEDIAN_DEFAULTS)
        output = _median(image_rgb, **values)
    elif method == "clahe+median":
        values = _take_params(
            params,
            {**_CLAHE_DEFAULTS, **_MEDIAN_DEFAULTS},
        )
        output = _clahe(
            image_rgb,
            clip_limit=values["clip_limit"],
            tile_grid_size=values["tile_grid_size"],
        )
        output = _median(output, median_ksize=values["median_ksize"])
    else:
        raise ValueError(
            f"unknown method {method!r}; choose one of: {', '.join(METHODS)}"
        )

    if output.shape != image_rgb.shape or output.dtype != np.uint8:
        raise RuntimeError("preprocessing changed the image shape or dtype")
    return output


def _main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser(
        description="Preprocess one image and save the result."
    )
    parser.add_argument("input", type=Path, help="source image")
    parser.add_argument("output", type=Path, help="destination image")
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=METHODS,
        help="preprocessing method (default: none)",
    )
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument(
        "--tile-grid-size",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("TILE_W", "TILE_H"),
    )
    parser.add_argument("--median-ksize", type=int, default=3)
    args = parser.parse_args()

    with Image.open(args.input) as opened:
        image_rgb = np.asarray(opened.convert("RGB"))

    params: dict[str, Any] = {}
    if args.method in ("clahe", "clahe+median"):
        params["clip_limit"] = args.clip_limit
        params["tile_grid_size"] = tuple(args.tile_grid_size)
    if args.method in ("median", "clahe+median"):
        params["median_ksize"] = args.median_ksize

    result = preprocess_image(image_rgb, args.method, **params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.output)
    print(f"Saved {args.method} result to {args.output}")


if __name__ == "__main__":
    _main()
