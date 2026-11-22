"""Simple, explainable image quality measurements."""
import cv2
import numpy as np

from .transforms import to_grayscale


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian. Many edges give a high value, so low values usually mean blur.

    The threshold depends on the scene, so compare values within one dataset only.
    """
    return float(cv2.Laplacian(to_grayscale(image), cv2.CV_64F).var())


def quality_report(image: np.ndarray) -> dict:
    """Summarise brightness, contrast, and sharpness of one image."""
    gray = to_grayscale(image)
    height, width = image.shape[:2]
    return {
        "height": int(height),
        "width": int(width),
        "mean_brightness": round(float(gray.mean()), 2),
        "contrast_std": round(float(gray.std()), 2),
        "sharpness_laplacian_var": round(sharpness(image), 2),
    }