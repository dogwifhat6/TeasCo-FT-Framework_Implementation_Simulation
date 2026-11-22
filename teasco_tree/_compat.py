"""
_compat.py — Native Runtime Load Order
=======================================

On macOS, PyTorch, XGBoost and LightGBM each link their own copy of the OpenMP
runtime. Whichever ``libomp.dylib`` is dlopen'd **first** wins for the process;
if PyTorch's copy loads first, XGBoost and LightGBM abort the moment they open a
parallel region::

    OMP: Error #179: Function pthread_mutex_init failed
    OMP: System error #22: Invalid argument

The failure is a hard abort, not an exception, so it cannot be caught and
retried — and it strikes at *train* time, long after the imports appear to have
succeeded. Neither ``KMP_DUPLICATE_LIB_OK=TRUE`` nor forcing single-threaded
boosting avoids it; only load order does.

:func:`preload_boosting_runtimes` therefore imports the boosting libraries
before anything pulls in torch. ``teasco_tree/__init__.py`` calls it as its very
first statement, so ``import teasco_tree`` is safe regardless of what the caller
does next.

The one arrangement that stays broken is importing ``torch`` *before*
``teasco_tree``. That is detected here and reported as a clear warning rather
than an unexplained abort thousands of trees later.

Set ``TEASCO_TREE_NO_PRELOAD=1`` to skip this entirely (e.g. MLP-only runs where
the import cost is unwanted).
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from typing import List

#: Libraries that must claim the OpenMP runtime before torch does.
_BOOSTING_MODULES = ("xgboost", "lightgbm")

_preloaded = False

#: Modules this process loaded *after* torch — using them risks a hard abort.
_unsafe: "set[str]" = set()


def preload_boosting_runtimes(warn_on_late: bool = True) -> List[str]:
    """
    Import the OpenMP-linked boosting libraries ahead of torch.

    Safe to call repeatedly; the work happens once.

    Parameters
    ----------
    warn_on_late : bool
        Warn if torch is already imported, meaning the ordering fix has been
        missed and tree adapters may abort the process during training.

    Returns
    -------
    list of str — the boosting modules successfully loaded.
    """
    global _preloaded

    if _preloaded or os.environ.get("TEASCO_TREE_NO_PRELOAD") == "1":
        return []

    torch_already_loaded = "torch" in sys.modules

    loaded: List[str] = []
    for module_name in _BOOSTING_MODULES:
        # Only our own late import is unsafe. If the module was already in
        # sys.modules the caller sequenced it themselves and we cannot know —
        # and should not second-guess — the order they achieved.
        already_present = module_name in sys.modules
        try:
            importlib.import_module(module_name)
        except Exception:
            # Not installed, or unusable on this platform. Tree adapters raise
            # their own actionable ImportError when actually requested.
            continue

        loaded.append(module_name)
        if torch_already_loaded and not already_present:
            _unsafe.add(module_name)

    if _unsafe and warn_on_late:
        warnings.warn(
            "torch was imported before teasco_tree, so PyTorch has already "
            f"claimed the OpenMP runtime. Training a tree adapter "
            f"({', '.join(sorted(_unsafe))}) would abort the process with "
            "'OMP: Error #179'. Import teasco_tree (or xgboost) before torch.",
            RuntimeWarning,
            stacklevel=2,
        )

    _preloaded = True
    return loaded


def check_boosting_safe(module_name: str) -> None:
    """
    Raise before a boosting library can abort the process.

    ``OMP: Error #179`` is a hard ``abort()``: it cannot be caught, it discards
    an in-flight benchmark, and on macOS it raises a "Python quit unexpectedly"
    crash dialog. Converting the condition into an ordinary exception means a
    misordered import costs one traceback instead of the whole run.

    Raises
    ------
    RuntimeError
        If ``module_name`` was loaded after torch in this process.
    """
    if module_name in _unsafe:
        raise RuntimeError(
            f"{module_name} was imported after torch in this process, so its "
            "OpenMP runtime conflicts with PyTorch's. Training would abort the "
            "process ('OMP: Error #179'), so it is refused here.\n"
            "Fix: import teasco_tree (or xgboost/lightgbm) before torch — in "
            "scripts, put `import teasco_tree` above any torch import; under "
            "pytest, tests/conftest.py already does this."
        )
