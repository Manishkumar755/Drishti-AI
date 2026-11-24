"""Classical computer vision: edges, contours, morphology, and motion.

Inputs are OpenCV images: BGR (height, width, 3) or grayscale (height, width).
"""
import cv2
import numpy as np

from ..preprocessing.transforms import to_grayscale


def canny_edges(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge detector. Keeps pixels where the gradient is strong and connected to strong edges."""
    return cv2.Canny(to_grayscale(image), low, high)


def sobel_magnitude(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Gradient magnitude from Sobel filters in x and y: sqrt(gx^2 + gy^2), scaled to 0..255."""
    gray = to_grayscale(image).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def laplacian_edges(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Second-derivative edges. Responds to edges from both sides, so it is noisier than Canny."""
    lap = cv2.Laplacian(to_grayscale(image), cv2.CV_64F, ksize=ksize)
    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Opening removes small specks, then closing fills small holes inside objects."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def find_objects(mask: np.ndarray, min_area: int = 500) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (x, y, w, h) of external contours whose area is at least min_area."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def frame_difference(prev_gray: np.ndarray, curr_gray: np.ndarray, threshold: int = 25) -> np.ndarray:
    """Pixels that changed by more than threshold between two consecutive frames."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def make_background_subtractor(history: int = 200, var_threshold: float = 16) -> cv2.BackgroundSubtractorMOG2:
    """MOG2 learns a per-pixel background model (a mixture of Gaussians) and flags pixels that do not fit it."""
    return cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=True)


def motion_mask(subtractor: cv2.BackgroundSubtractorMOG2, frame: np.ndarray) -> np.ndarray:
    """Foreground mask with shadows removed. MOG2 marks shadows as 127, so keep only values above 200."""
    fg = subtractor.apply(frame)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    return fg


def dense_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Farneback dense optical flow. Returns (height, width, 2) with per-pixel dx and dy in pixels."""
    return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)


def mean_flow(flow: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """Average (dx, dy) over the whole frame, or only over pixels where mask is set."""
    dx = flow[..., 0]
    dy = flow[..., 1]
    if mask is not None:
        selected = mask > 0
        dx = dx[selected]
        dy = dy[selected]
    if dx.size == 0:
        return 0.0, 0.0
    return float(dx.mean()), float(dy.mean())