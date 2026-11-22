import numpy as np
import pytest

from cv_engine.preprocessing import transforms as t
from cv_engine.preprocessing.quality import quality_report


@pytest.fixture
def pure_red() -> np.ndarray:
    """A 120x160 BGR image where every pixel is red (B=0, G=0, R=200)."""
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :, 2] = 200  # index 2 is red in BGR order
    return image


def test_grayscale_drops_channel_dimension(pure_red):
    assert t.to_grayscale(pure_red).shape == (120, 160)


def test_grayscale_of_red_is_about_60(pure_red):
    # Luminance = 0.299 R + 0.587 G + 0.114 B, so 0.299 * 200 is about 60.
    gray_mean = t.to_grayscale(pure_red).mean()
    assert 55 <= gray_mean <= 65


def test_resize_caps_long_side_and_keeps_aspect():
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    out = t.resize_long_side(image, max_side=640)
    assert max(out.shape[:2]) == 640
    assert out.shape[1] / out.shape[0] == pytest.approx(2.0, rel=0.02)


def test_resize_never_enlarges_small_images():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert t.resize_long_side(image, max_side=640).shape == image.shape


def test_otsu_output_is_strictly_binary():
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, size=(50, 50), dtype=np.uint8)
    out = t.otsu_threshold(gray)
    assert set(np.unique(out)) <= {0, 255}


def test_clahe_keeps_shape_and_dtype(pure_red):
    out = t.apply_clahe(pure_red)
    assert out.shape == pure_red.shape
    assert out.dtype == np.uint8


def test_quality_report_has_expected_keys(pure_red):
    report = quality_report(pure_red)
    assert set(report) == {"height", "width", "mean_brightness",
                           "contrast_std", "sharpness_laplacian_var"}
    assert report["height"] == 120 and report["width"] == 160