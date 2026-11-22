"""
teasco_tree — TeasCo-FT with gradient-boosted tree adapters.

    frozen Student SLM (local Ollama)  ─┐
                                        ├─> 385-dim features ─> tree adapter
    sentence embedding (MiniLM, 384)   ─┘                        (XGBoost / CatBoost / LightGBM)
                                                                  ^
    70B Teacher LLM (offline, once) ── soft labels ───────────────┘

No MLP anywhere: the only trainable part is a boosted-tree ensemble.
"""

# MUST run before anything imports torch (sentence-transformers does) — see
# teasco_tree/_compat.py. Keep this the first statement of the package.
from teasco_tree._compat import preload_boosting_runtimes as _preload

_preload()

from teasco_tree.adapters import available_adapters, get_adapter  # noqa: E402

__version__ = "1.0.0"
__all__ = ["available_adapters", "get_adapter", "__version__"]
