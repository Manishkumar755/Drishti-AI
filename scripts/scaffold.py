"""Create the Drishti-AI directory layout.

Idempotent: safe to run more than once. Existing files are never overwritten.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRECTORIES = [
    "backend/app/api",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/db",
    "backend/cv_engine/io",
    "backend/cv_engine/preprocessing",
    "backend/cv_engine/detection",
    "backend/cv_engine/tracking",
    "backend/cv_engine/events",
    "backend/cv_engine/analytics",
    "backend/cv_engine/ml",
    "backend/cv_engine/segmentation",
    "backend/cv_engine/pose",
    "backend/cv_engine/ocr",
    "backend/configs",
    "backend/tests",
    "frontend",
    "scripts",
    "data/raw",
    "data/processed",
    "data/samples",
    "models",
    "notebooks",
    "docs/decisions",
    "results",
]

# Directories that are Python packages get an __init__.py
PACKAGE_ROOTS = ["backend/app", "backend/cv_engine"]


def main() -> None:
    for rel in DIRECTORIES:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)

    for pkg_root in PACKAGE_ROOTS:
        base = ROOT / pkg_root
        for folder in [base, *base.rglob("*")]:
            if folder.is_dir():
                init_file = folder / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("", encoding="utf-8")

    for keep in ["data/raw", "data/processed", "data/samples", "models", "results"]:
        keep_file = ROOT / keep / ".gitkeep"
        if not keep_file.exists():
            keep_file.write_text("", encoding="utf-8")

    print(f"Scaffold created under {ROOT}")


if __name__ == "__main__":
    main()