"""Phase 2 demo: save a grid comparing preprocessing steps on one image.

If data/samples/road.jpg does not exist, a synthetic road scene is created
so the demo works offline. You can replace it with your own photo later.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from cv_engine.preprocessing import transforms as t  # noqa: E402  # type: ignore[reportMissingImports]
from cv_engine.preprocessing.quality import quality_report  # noqa: E402  # type: ignore[reportMissingImports]

SAMPLE = ROOT / "data" / "samples" / "road.jpg"
OUTPUT = ROOT / "results" / "preprocess_comparison.png"
TILE_W, TILE_H = 320, 240


def make_synthetic_road(height: int = 480, width: int = 640) -> np.ndarray:
    """Draw a simple road scene and add Gaussian noise."""
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    image[: height // 3] = (200, 170, 120)
    cv2.rectangle(image, (0, height // 3), (width, height), (80, 80, 80), -1)
    for x in range(40, width, 120):
        cv2.rectangle(image, (x, height * 3 // 4),
                      (x + 50, height * 3 // 4 + 8), (255, 255, 255), -1)
    cv2.rectangle(image, (250, 250), (400, 330), (40, 60, 200), -1)
    cv2.circle(image, (270, 335), 14, (20, 20, 20), -1)
    cv2.circle(image, (380, 335), 14, (20, 20, 20), -1)
    noise = np.random.default_rng(0).normal(0, 12, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def make_tile(image: np.ndarray, label: str) -> np.ndarray:
    """Resize to a fixed tile, convert to BGR for stacking, and add a label bar."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(image, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (TILE_W, 22), (0, 0, 0), -1)
    cv2.putText(tile, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)
    return tile


def load_or_create_sample() -> tuple[np.ndarray, str]:
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    if SAMPLE.exists():
        image = cv2.imread(str(SAMPLE))
        if image is None:
            raise SystemExit(f"OpenCV could not read {SAMPLE}")
        return image, SAMPLE.name
    image = make_synthetic_road()
    cv2.imwrite(str(SAMPLE), image)
    return image, "synthetic road (saved to data/samples/road.jpg)"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image, source = load_or_create_sample()
    image = t.resize_long_side(image, max_side=640)
    gray = t.to_grayscale(image)
    clahe_image = t.apply_clahe(image)

    variants = [
        ("1 original", image),
        ("2 grayscale", gray),
        ("3 gaussian 5x5", t.gaussian_blur(image, 5)),
        ("4 median 5x5", t.median_blur(image, 5)),
        ("5 bilateral", t.bilateral_filter(image)),
        ("6 CLAHE (LAB L)", clahe_image),
        ("7 sharpen", t.sharpen(image)),
        ("8 otsu binary", t.otsu_threshold(gray)),
    ]
    tiles = [make_tile(img, label) for label, img in variants]
    grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(OUTPUT), grid)

    print(f"Source: {source}")
    print(f"Saved comparison grid to {OUTPUT.relative_to(ROOT)}")
    print("\nQuality before and after CLAHE:")
    print("  original:", quality_report(image))
    print("  CLAHE:   ", quality_report(clahe_image))


if __name__ == "__main__":
    main()