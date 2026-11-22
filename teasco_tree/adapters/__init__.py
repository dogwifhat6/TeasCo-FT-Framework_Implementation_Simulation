"""
Tree adapters — the only trainable part of the cascade.

    from teasco_tree.adapters import get_adapter

    adapter = get_adapter("xgboost", classes=["sec", "nonsec"])
    adapter.fit(X_train, y_train, X_val, y_val)
    adapter.evaluate(X_test, y_test)["macro_f1"]

Deliberately tree-only: there is no MLP adapter in this project.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

from teasco_tree.adapters.base import BaseAdapter

#: Adapter name -> class name in tree_adapters.py.
_REGISTRY = {
    "xgboost": "XGBoostAdapter",
    "catboost": "CatBoostAdapter",
    "lightgbm": "LightGBMAdapter",
}

_ALIASES = {"xgb": "xgboost", "cat": "catboost", "lgbm": "lightgbm", "lgb": "lightgbm"}

#: Gate modes each library supports (see tree_adapters.py for what they mean).
GATE_MODES: Dict[str, List[str]] = {
    "xgboost": ["objective", "staged", "none"],
    "catboost": ["staged", "none"],
    "lightgbm": ["staged", "none"],
}

#: Default gate mode per adapter.
DEFAULT_GATE_MODE = {"xgboost": "objective", "catboost": "staged", "lightgbm": "staged"}

DISPLAY_NAME = {"xgboost": "XGBoost", "catboost": "CatBoost", "lightgbm": "LightGBM"}


def available_adapters() -> List[str]:
    """Names accepted by :func:`get_adapter`, in display order."""
    return list(_REGISTRY)


def resolve_name(name: str) -> str:
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in _REGISTRY:
        raise ValueError(f"Unknown adapter {name!r}. Available: {available_adapters()}")
    return key


def get_adapter(name: str, classes: List[str], **kwargs: Any) -> BaseAdapter:
    """Instantiate a tree adapter by name (aliases such as ``"xgb"`` accepted)."""
    key = resolve_name(name)
    module = importlib.import_module("teasco_tree.adapters.tree_adapters")
    return getattr(module, _REGISTRY[key])(classes=classes, **kwargs)


__all__ = [
    "BaseAdapter", "DEFAULT_GATE_MODE", "DISPLAY_NAME", "GATE_MODES",
    "available_adapters", "get_adapter", "resolve_name",
]
