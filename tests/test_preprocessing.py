"""Unit tests for src/preprocessing.py.

Run from the repository root::

    python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import available_methods, preprocess_image


def _image(shape: tuple[int, int, int] = (48, 64, 3), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def test_expected_methods_are_available() -> None:
    assert set(available_methods()) == {
        "none",
        "clahe",
        "median",
        "clahe+median",
    }


@pytest.mark.parametrize("method", ["none", "clahe", "median", "clahe+median"])
def test_method_keeps_shape_and_dtype(method: str) -> None:
    image = _image()
    output = preprocess_image(image, method)
    assert output.shape == image.shape
    assert output.dtype == np.uint8


def test_none_returns_a_copy() -> None:
    image = _image()
    output = preprocess_image(image, "none")
    assert np.array_equal(output, image)
    assert output is not image


@pytest.mark.parametrize("method", ["none", "clahe", "median", "clahe+median"])
def test_input_array_is_not_modified(method: str) -> None:
    image = _image()
    original = image.copy()
    preprocess_image(image, method)
    assert np.array_equal(image, original)


def test_median_reduces_salt_and_pepper_noise() -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    image[::4, ::4] = 255
    output = preprocess_image(image, "median", median_ksize=3)
    assert output.std() < image.std()


def test_clahe_keeps_red_dominant() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = 180
    image[..., 1] = 40
    image[..., 2] = 40
    output = preprocess_image(image, "clahe")
    assert output[..., 0].mean() > output[..., 2].mean()


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
    ],
)
def test_invalid_image_is_rejected(bad_image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        preprocess_image(bad_image, "clahe")


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        preprocess_image(_image(), "sharpen")


def test_unknown_parameter_is_rejected() -> None:
    with pytest.raises(TypeError):
        preprocess_image(_image(), "clahe", clipLimit=3.0)


def test_even_median_kernel_is_rejected() -> None:
    with pytest.raises(ValueError):
        preprocess_image(_image(), "median", median_ksize=4)
