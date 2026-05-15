"""Quick environment check for the helmet detection project."""

from __future__ import annotations

import importlib.util
import sys


PACKAGES = [
    "cv2",
    "numpy",
    "torch",
    "ultralytics",
    "yaml",
]


def main():
    missing = [pkg for pkg in PACKAGES if importlib.util.find_spec(pkg) is None]
    print(f"Python: {sys.version.split()[0]}")
    if missing:
        print("Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall dependencies with: pip install -r requirements.txt")
        raise SystemExit(1)

    import torch
    import ultralytics

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print("Environment looks ready.")


if __name__ == "__main__":
    main()
