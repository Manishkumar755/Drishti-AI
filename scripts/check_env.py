"""Verify that the core toolchain for Drishti-AI is installed and working.

Exit code 0 means every check passed. Exit code 1 means something is missing.
"""
import importlib
import platform
import sys

REQUIRED_MODULES = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "yaml": "pyyaml",
    "torch": "torch",
    "torchvision": "torchvision",
    "sklearn": "scikit-learn",
    "ultralytics": "ultralytics",
}


def check_python() -> bool:
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro} "
          f"on {platform.system()} {platform.machine()}")
    if not ((3, 10) <= (version.major, version.minor) <= (3, 12)):
        print("  [WARN] Python 3.10 to 3.12 is recommended for torch and ultralytics wheels.")
        return False
    return True


def check_modules() -> list[str]:
    failures = []
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "version unknown")
            print(f"  [OK]   {package_name:<14} {version}")
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            print(f"  [FAIL] {package_name:<14} {type(exc).__name__}: {exc}")
            failures.append(package_name)
    return failures


def check_opencv_smoke_test() -> bool:
    import cv2
    import numpy as np

    # A 64x64 BGR image: 3 channels, uint8, 0 to 255
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # set the red channel (index 2 in BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ok = image.shape == (64, 64, 3) and gray.shape == (64, 64)
    print(f"  [{'OK' if ok else 'FAIL'}]   OpenCV array round-trip "
          f"(shape {image.shape} -> gray {gray.shape})")
    return ok


def check_torch() -> None:
    import torch

    cuda = torch.cuda.is_available()
    print(f"  [INFO] torch {torch.__version__}, CUDA available: {cuda}")
    print("         CPU is sufficient for Phases 1 through 6.")


def main() -> int:
    print("== Python ==")
    python_ok = check_python()

    print("\n== Packages ==")
    failures = check_modules()

    if failures:
        print(f"\nMissing or broken: {', '.join(failures)}")
        print("Fix: activate your .venv, then run: pip install -r requirements.txt")
        return 1

    print("\n== Checks ==")
    check_torch()
    cv_ok = check_opencv_smoke_test()

    print()
    if python_ok and cv_ok:
        print("ENVIRONMENT OK")
        return 0
    print("ENVIRONMENT HAS WARNINGS. Read the lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())