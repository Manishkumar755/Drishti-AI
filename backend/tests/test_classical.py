import cv2
import numpy as np
import pytest

from cv_engine.detection import classical as c


@pytest.fixture
def white_square_on_black() -> np.ndarray:
    """200x200 BGR image with a filled white square from (50,50) to (150,150)."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
    return image


def test_canny_detects_square_border(white_square_on_black):
    edges = c.canny_edges(white_square_on_black)
    assert edges.shape == (200, 200)
    assert np.count_nonzero(edges) > 100


def test_sobel_output_shape_and_dtype(white_square_on_black):
    out = c.sobel_magnitude(white_square_on_black)
    assert out.shape == (200, 200)
    assert out.dtype == np.uint8
    assert out.max() == 255


def test_laplacian_output_shape_and_dtype(white_square_on_black):
    out = c.laplacian_edges(white_square_on_black)
    assert out.shape == (200, 200)
    assert out.dtype == np.uint8


def test_find_objects_returns_large_box():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[40:120, 30:130] = 255  # 80 tall, 100 wide
    boxes = c.find_objects(mask, min_area=500)
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert (w, h) == (100, 80)


def test_find_objects_ignores_small_noise():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[10:15, 10:15] = 255  # 25 pixels, below min_area
    assert c.find_objects(mask, min_area=500) == []


def test_clean_mask_removes_isolated_pixel():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10, 10] = 255          # single speck
    mask[20:60, 20:60] = 255    # real object
    cleaned = c.clean_mask(mask, kernel_size=5)
    assert cleaned[10, 10] == 0
    assert cleaned[40, 40] == 255


def test_frame_difference_detects_moved_square():
    prev = np.zeros((100, 100), dtype=np.uint8)
    curr = np.zeros((100, 100), dtype=np.uint8)
    prev[20:40, 10:30] = 200
    curr[20:40, 30:50] = 200
    mask = c.frame_difference(prev, curr, threshold=25)
    assert np.count_nonzero(mask) > 0
    assert mask[30, 35] == 255  # newly covered area


def test_flow_detects_rightward_shift():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 256, (64, 64), dtype=np.uint8), (5, 5), 0)
    shifted = np.roll(base, 2, axis=1)  # move content 2 px to the right
    flow = c.dense_flow(base, shifted)
    dx, _ = c.mean_flow(flow)
    assert dx > 0.5