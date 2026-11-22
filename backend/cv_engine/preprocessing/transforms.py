"""Basic image operations.

All functions take and return images in OpenCV's default format:
BGR, uint8, shape (height, width, channels).
"""
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to a single-channel grayscale image."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_long_side(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Shrink the image so its longer side is at most max_side. Never enlarges."""
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return image
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Smooth with a Gaussian kernel. Reduces Gaussian-like noise."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Replace each pixel with the median of its neighbourhood. Good for salt-and-pepper noise."""
    return cv2.medianBlur(image, ksize)


def bilateral_filter(image: np.ndarray, d: int = 9,
                     sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Smooth flat regions while keeping edges sharp."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0,
                               beta: float = 0.0) -> np.ndarray:
    """output = alpha * input + beta, clipped to 0..255."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Apply a 3x3 sharpening kernel. Boosts edges and amplifies noise."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on the LAB lightness channel.

    Working on L alone improves contrast without shifting colours.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_grid, tile_grid))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)),
                        cv2.COLOR_LAB2BGR)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Local threshold: each pixel is compared with its neighbourhood. Handles uneven lighting."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Global threshold chosen automatically by Otsu's method (minimises intra-class variance)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary