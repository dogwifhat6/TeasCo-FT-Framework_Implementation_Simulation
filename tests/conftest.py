"""
pytest setup. Import the package before anything can pull in torch: PyTorch,
XGBoost and LightGBM each ship their own libomp, and if torch's loads first the
boosting libraries abort the interpreter (see teasco_tree/_compat.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import teasco_tree  # noqa: E402,F401
