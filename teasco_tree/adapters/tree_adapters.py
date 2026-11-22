"""
tree_adapters.py — Gradient-Boosted Tree Adapters (XGBoost / CatBoost / LightGBM)
=================================================================================

Drop-in replacements for the MLP adapter. They consume the same 385-dim
representation and honour the same gated update rule, but replace backprop with
boosting.

Gate modes
----------
``objective``
    The exact analogue of :class:`GatedLoss`. A custom softmax objective returns
    ``grad=0, hess=eps`` for every sample the ensemble already classifies
    correctly, so a new tree can only fit the errors. Because boosting recomputes
    margins each round, the gate is re-evaluated each round — matching the MLP's
    per-mini-batch re-gating. XGBoost only (it is the one library here exposing a
    clean custom multiclass objective).

``staged``
    Library-agnostic. Training proceeds in stages; before each stage the current
    ensemble predicts on the training set and correct samples are down-weighted
    to ``gate_floor`` (0.0 = strict). The next block of trees is warm-started
    from the existing ensemble via ``xgb_model``/``init_model``. Granularity is
    per-stage rather than per-round, analogous to gating per-epoch.

``none``
    Ungated control. Boosting already concentrates on residuals, so this
    measures whether the *explicit* gate contributes anything beyond what
    boosting does natively — without it, any gain claimed for the gate is
    unfalsifiable.

All three modes share one training skeleton: fit a block of trees, score
validation macro F1, keep the best ensemble size, stop after ``patience``
stages without improvement. Rather than snapshotting boosters, the best size is
recorded and applied at prediction time (``iteration_range`` / ``num_iteration``
/ ``ntree_end``), which is cheaper and avoids deep-copying large models.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from teasco_tree._compat import check_boosting_safe
from teasco_tree.adapters.base import BaseAdapter
from teasco_tree.adapters.gated_objective import (
    gate_sample_weights,
    make_xgboost_gated_objective,
    softmax,
)

logger = logging.getLogger(__name__)

GATE_MODES = ("objective", "staged", "none")


def _balanced_weights(
    y: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    sklearn's "balanced" weighting, computed over weighted class *mass*.

    With plain labels this is ``compute_sample_weight("balanced", y)``. With
    soft-label expansion a class's size is the sum of the Teacher probability
    assigned to it, not its row count — every class appears once per text, so
    counting rows would call every task perfectly balanced.
    """
    if sample_weight is None:
        return compute_sample_weight("balanced", y)
    classes = np.unique(y)
    mass = np.array([sample_weight[y == c].sum() for c in classes])
    per_class = sample_weight.sum() / (len(classes) * np.maximum(mass, 1e-12))
    lookup = dict(zip(classes, per_class))
    return sample_weight * np.array([lookup[c] for c in y])


class GatedBoostingAdapter(BaseAdapter):
    """
    Shared training skeleton for gated gradient-boosted adapters.

    Parameters
    ----------
    classes : list of str
        Ordered class names.
    gate_mode : {"objective", "staged", "none"}
        How the gated update rule is realised (see module docstring).
    n_estimators : int
        Maximum total trees (per class, for multiclass). Default: 400.
    rounds_per_stage : int
        Trees added between gate refreshes / validation checks. Default: 25.
    patience : int
        Stages without validation improvement before stopping. Default: 6.
    learning_rate, max_depth, subsample, colsample, reg_lambda
        Standard boosting hyper-parameters.
    gate_floor : float
        Weight kept by already-correct samples in ``staged`` mode. Default: 0.0.
    class_weight : {"balanced", None}
        Balance classes via sample weights — matters for the skewed security
        and 12-class tasks. Default: None.
    label_smoothing : float
        Applied in ``objective`` mode to mirror the MLP's loss. Default: 0.05.
    gate_margin : float
        Confidence band protecting the ``objective`` gate from collapse. A
        strict correctness gate (0.0) silences an entire class on the first
        round — boosting starts from a constant margin, so every sample ties —
        leaving no contrast to split on and yielding single-leaf trees forever.
        Measured on PROMISE/binary: ``gate_margin=0`` produces 0 splits and
        macro F1 0.38 (a constant model); ``0.10`` produces 222 splits and 0.72.
        Default: 0.10.

        The default needs no warmup because it is self-warming: at
        initialisation the top-two probability gap is exactly 0, so nothing is
        silenced and the first rounds train ungated automatically.
    warmup_rounds : int
        Extra ungated rounds before the gate engages, available for datasets
        where the margin alone leaves the early ensemble degenerate.
        Default: 0.
    """

    family = "tree"

    def __init__(
        self,
        classes: List[str],
        gate_mode: str = "staged",
        n_estimators: int = 400,
        rounds_per_stage: int = 25,
        patience: int = 6,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.9,
        colsample: float = 0.8,
        reg_lambda: float = 1.0,
        gate_floor: float = 0.0,
        class_weight: Optional[str] = None,
        label_smoothing: float = 0.05,
        gate_margin: float = 0.10,
        warmup_rounds: int = 0,
        random_state: int = 42,
        verbose: bool = True,
        **extra_params: Any,
    ):
        super().__init__(classes=classes, random_state=random_state, verbose=verbose)

        if gate_mode not in GATE_MODES:
            raise ValueError(
                f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}."
            )

        self.gate_mode = gate_mode
        self.n_estimators = n_estimators
        self.rounds_per_stage = rounds_per_stage
        self.patience = patience
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample = colsample
        self.reg_lambda = reg_lambda
        self.gate_floor = gate_floor
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.gate_margin = gate_margin
        self.warmup_rounds = warmup_rounds
        self.extra_params = extra_params

        self.n_stages = max(1, n_estimators // rounds_per_stage)
        self.best_n_trees: int = 0
        self._model = None

    # ── Hooks implemented per library ──────────────────────────────────────

    def _fit_stage(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray],
        n_rounds: int,
        first_stage: bool,
    ) -> None:
        """Add ``n_rounds`` trees, warm-starting from the current ensemble."""
        raise NotImplementedError

    def _raw_predict_proba(
        self, X: np.ndarray, n_trees: Optional[int] = None
    ) -> np.ndarray:
        """Probabilities using only the first ``n_trees`` trees."""
        raise NotImplementedError

    @property
    def _current_n_trees(self) -> int:
        raise NotImplementedError

    # ── Shared training loop ───────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        gate_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        ``sample_weight`` is how soft-label distillation reaches a tree: the
        pipeline expands every row into one copy per class weighted by the
        Teacher's probability for that class (see
        :func:`teasco_tree.pipeline.expand_soft_labels`). It multiplies into the
        class balancing and, in ``staged`` mode, into the gate.

        ``gate_labels`` goes with it: the per-row label of the *source text*
        (the Teacher's argmax). The gate and the reported train F1 judge
        correctness against it, since an expanded copy's own label is by
        construction wrong for all but one copy of each text.
        """
        t_start = time.time()

        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        has_val = X_val is not None and y_val is not None
        gate_y = y if gate_labels is None else np.asarray(gate_labels, dtype=np.int64)
        self._gate_labels = None if gate_labels is None else gate_y

        base_weight = None
        if sample_weight is not None:
            base_weight = np.asarray(sample_weight, dtype=np.float64)
        if self.class_weight == "balanced":
            base_weight = _balanced_weights(y, base_weight)

        self.history = {
            "train_macro_f1": [],
            "val_macro_f1": [],
            "gate_ratio": [],
            "n_trees": [],
        }

        best_val_f1, best_stage, patience_counter = 0.0, 0, 0
        self.best_n_trees = 0
        stage = 0

        # Predictions scored at the end of a stage describe exactly the
        # ensemble the next stage gates against, so they are carried forward
        # instead of being recomputed over the whole training set.
        train_proba: Optional[np.ndarray] = None

        for stage in range(1, self.n_stages + 1):
            first = stage == 1

            # ── Compute the gate from the *current* ensemble ───────────────
            if self.gate_mode == "staged" and not first:
                train_preds = train_proba.argmax(axis=1)
                weights, n_incorrect = gate_sample_weights(
                    train_preds, gate_y, gate_floor=self.gate_floor,
                    base_weight=base_weight,
                )
                gate_ratio = n_incorrect / len(y)

                # Mirrors the MLP: nothing wrong means nothing to learn from.
                if n_incorrect == 0:
                    if self.verbose:
                        logger.info(
                            "Stage %d: training set fully separated — stopping.",
                            stage,
                        )
                    break
            else:
                weights = base_weight
                gate_ratio = 1.0

            self._fit_stage(X, y, weights, self.rounds_per_stage, first)

            # ── Score this stage (result feeds the next stage's gate) ─────
            train_proba = self._raw_predict_proba(X)
            train_f1 = f1_score(
                gate_y, train_proba.argmax(axis=1),
                average="macro", zero_division=0,
            )
            val_f1 = 0.0
            if has_val:
                val_f1 = f1_score(
                    y_val, self._raw_predict_proba(X_val).argmax(axis=1),
                    average="macro", zero_division=0,
                )

            n_trees = self._current_n_trees
            self.history["train_macro_f1"].append(float(train_f1))
            self.history["val_macro_f1"].append(float(val_f1))
            self.history["gate_ratio"].append(float(gate_ratio))
            self.history["n_trees"].append(int(n_trees))

            if self.verbose:
                logger.info(
                    "Stage %2d/%d | trees %4d | gate %.1f%% | train F1 %.4f | val F1 %.4f",
                    stage, self.n_stages, n_trees, 100 * gate_ratio,
                    train_f1, val_f1,
                )

            monitored = val_f1 if has_val else train_f1
            if monitored > best_val_f1:
                best_val_f1, best_stage = monitored, stage
                self.best_n_trees = n_trees
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        logger.info("Early stopping at stage %d", stage)
                    break

        if self.best_n_trees == 0:
            self.best_n_trees = self._current_n_trees

        self._fitted = True
        self.best_val_f1 = float(best_val_f1)
        self.best_iteration = best_stage

        return {
            "adapter": self.name,
            "gate_mode": self.gate_mode,
            "best_val_f1": self.best_val_f1,
            "best_iteration": best_stage,
            "best_n_trees": self.best_n_trees,
            "total_stages": stage,
            "elapsed_seconds": time.time() - t_start,
            "history": self.history,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self._raw_predict_proba(X, n_trees=self.best_n_trees)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostAdapter(GatedBoostingAdapter):
    """
    XGBoost adapter. The only adapter supporting ``gate_mode="objective"``,
    the exact gradient-level analogue of :class:`GatedLoss`.
    """

    name = "xgboost"

    def __init__(self, classes: List[str], gate_mode: str = "objective", **kwargs: Any):
        super().__init__(classes=classes, gate_mode=gate_mode, **kwargs)
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "xgboost is required for XGBoostAdapter. "
                "Install it with: pip install xgboost"
            ) from e
        # Fail loudly now rather than aborting the process mid-training.
        check_boosting_safe("xgboost")
        self._xgb = xgb
        self._booster = None

    def _params(self) -> Dict[str, Any]:
        params = {
            "objective": "multi:softprob",
            "num_class": self.num_classes,
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample,
            "lambda": self.reg_lambda,
            "seed": self.random_state,
            "nthread": -1,
            # A custom objective has no matching built-in metric.
            "disable_default_eval_metric": 1,
        }
        params.update(self.extra_params)
        return params

    def _fit_stage(self, X, y, weights, n_rounds, first_stage):
        dtrain = self._xgb.DMatrix(X, label=y, weight=weights)

        obj = None
        if self.gate_mode == "objective":
            # Weights are folded into the objective instead of the DMatrix so
            # the gate and class balancing compose cleanly. Warmup is counted
            # across the whole ensemble, not restarted per stage.
            obj = make_xgboost_gated_objective(
                num_classes=self.num_classes,
                gate=True,
                label_smoothing=self.label_smoothing,
                sample_weight=weights,
                gate_margin=self.gate_margin,
                warmup_rounds=self.warmup_rounds,
                rounds_completed=self._current_n_trees,
                gate_labels=getattr(self, "_gate_labels", None),
            )

        self._booster = self._xgb.train(
            self._params(),
            dtrain,
            num_boost_round=n_rounds,
            xgb_model=None if first_stage else self._booster,
            obj=obj,
            verbose_eval=False,
        )

    def _raw_predict_proba(self, X, n_trees=None):
        if self._booster is None:
            raise RuntimeError("XGBoost booster has not been trained yet.")
        dm = self._xgb.DMatrix(X)
        iteration_range = (0, n_trees) if n_trees else (0, 0)
        # Always take raw margins and apply softmax here: with a custom
        # objective XGBoost cannot know the correct output transform.
        margins = self._booster.predict(
            dm, output_margin=True, iteration_range=iteration_range
        )
        margins = np.asarray(margins, dtype=np.float64)
        if margins.ndim == 1:
            margins = margins.reshape(-1, self.num_classes)
        return softmax(margins)

    @property
    def _current_n_trees(self) -> int:
        return 0 if self._booster is None else self._booster.num_boosted_rounds()

    def save(self, path: str) -> None:
        self._check_fitted()
        self._booster.save_model(path)

    def load(self, path: str, best_n_trees: Optional[int] = None) -> "XGBoostAdapter":
        self._booster = self._xgb.Booster()
        self._booster.load_model(path)
        # The file holds every tree trained, including the ones past the early-
        # stopping point; without best_n_trees a reload predicts with all of them.
        self.best_n_trees = best_n_trees or self._booster.num_boosted_rounds()
        self._fitted = True
        return self


# ─────────────────────────────────────────────────────────────────────────────
# CatBoost
# ─────────────────────────────────────────────────────────────────────────────

class CatBoostAdapter(GatedBoostingAdapter):
    """
    CatBoost adapter using staged gating.

    Stages are chained through CatBoost's ``baseline`` — each stage is fitted on
    a ``Pool`` whose baseline is the current ensemble's raw margin, then merged
    into the ensemble with :func:`catboost.sum_models`. It deliberately does
    **not** use ``fit(init_model=...)``: measured on CatBoost 1.2.10, the first
    continuation from a given model silently ignores ``sample_weight``, and
    repeating the same continuation gives a different model. Because every stage
    here continues exactly once, the gate was a no-op under ``init_model`` —
    ``staged`` and ``none`` produced byte-identical ensembles.
    ``tests/test_adapters.py::TestCatBoostGate`` pins this.
    """

    name = "catboost"

    def __init__(self, classes: List[str], gate_mode: str = "staged", **kwargs: Any):
        if gate_mode == "objective":
            logger.warning(
                "CatBoost does not expose a NumPy multiclass custom objective; "
                "falling back to gate_mode='staged'."
            )
            gate_mode = "staged"
        super().__init__(classes=classes, gate_mode=gate_mode, **kwargs)
        try:
            import catboost
        except ImportError as e:
            raise ImportError(
                "catboost is required for CatBoostAdapter. "
                "Install it with: pip install catboost"
            ) from e
        self._catboost = catboost
        self._model = None

    def _params(self) -> Dict[str, Any]:
        params = {
            "learning_rate": self.learning_rate,
            "depth": min(self.max_depth, 16),
            "l2_leaf_reg": self.reg_lambda,
            "loss_function": "MultiClass" if self.num_classes > 2 else "Logloss",
            "random_seed": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
        }
        params.update(self.extra_params)
        return params

    def _raw_margin(self, X, n_trees=None) -> np.ndarray:
        return np.asarray(self._model.predict(
            X, prediction_type="RawFormulaVal", ntree_end=n_trees or 0,
        ), dtype=np.float64)

    def _fit_stage(self, X, y, weights, n_rounds, first_stage):
        # CatBoost rejects an all-zero weight vector; the caller already breaks
        # when the gate closes completely, this guards the degenerate remainder.
        if weights is not None and float(np.sum(weights)) <= 0:
            return

        baseline = None if first_stage or self._model is None else self._raw_margin(X)
        pool = self._catboost.Pool(X, y, weight=weights, baseline=baseline)
        stage_model = self._catboost.CatBoostClassifier(iterations=n_rounds, **self._params())
        stage_model.fit(pool)

        if baseline is None:
            self._model = stage_model
        else:
            self._model = self._catboost.sum_models([self._model, stage_model])

    def _raw_predict_proba(self, X, n_trees=None):
        if self._model is None:
            raise RuntimeError("CatBoost model has not been trained yet.")
        margin = self._raw_margin(X, n_trees)
        if self.num_classes == 2:
            p1 = 1.0 / (1.0 + np.exp(-margin.reshape(-1)))
            return np.column_stack([1.0 - p1, p1])
        return softmax(margin.reshape(-1, self.num_classes))

    @property
    def _current_n_trees(self) -> int:
        return 0 if self._model is None else int(self._model.tree_count_)

    def save(self, path: str) -> None:
        self._check_fitted()
        self._model.save_model(path)

    def load(self, path: str, best_n_trees: Optional[int] = None) -> "CatBoostAdapter":
        self._model = self._catboost.CatBoost()
        self._model.load_model(path)
        self.best_n_trees = best_n_trees or int(self._model.tree_count_)
        self._fitted = True
        return self


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMAdapter(GatedBoostingAdapter):
    """LightGBM adapter using staged gating (``init_model`` warm start)."""

    name = "lightgbm"

    def __init__(self, classes: List[str], gate_mode: str = "staged", **kwargs: Any):
        if gate_mode == "objective":
            logger.warning(
                "LightGBM multiclass custom objectives use a class-major "
                "layout; falling back to gate_mode='staged'."
            )
            gate_mode = "staged"
        super().__init__(classes=classes, gate_mode=gate_mode, **kwargs)
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "lightgbm is required for LightGBMAdapter. "
                "Install it with: pip install lightgbm"
            ) from e
        # Fail loudly now rather than aborting the process mid-training.
        check_boosting_safe("lightgbm")
        self._lgb = lgb
        self._booster = None

    def _fit_stage(self, X, y, weights, n_rounds, first_stage):
        params = {
            "objective": "multiclass",
            "num_class": self.num_classes,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": min(2 ** self.max_depth, 63),
            "feature_fraction": self.colsample,
            "bagging_fraction": self.subsample,
            "bagging_freq": 1,
            "lambda_l2": self.reg_lambda,
            "seed": self.random_state,
            "verbose": -1,
            # Embedding features are dense and small-N; keep leaves reachable.
            "min_data_in_leaf": 5,
        }
        params.update(self.extra_params)

        if weights is not None and float(np.sum(weights)) <= 0:
            return

        dtrain = self._lgb.Dataset(X, label=y, weight=weights, free_raw_data=False)
        self._booster = self._lgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            init_model=None if first_stage else self._booster,
        )

    def _raw_predict_proba(self, X, n_trees=None):
        if self._booster is None:
            raise RuntimeError("LightGBM booster has not been trained yet.")
        kwargs = {"num_iteration": n_trees} if n_trees else {}
        proba = self._booster.predict(X, **kwargs)
        return np.asarray(proba, dtype=np.float64)

    @property
    def _current_n_trees(self) -> int:
        if self._booster is None:
            return 0
        # LightGBM counts one tree per class per round for multiclass.
        return self._booster.current_iteration()

    def save(self, path: str) -> None:
        self._check_fitted()
        self._booster.save_model(path, num_iteration=self.best_n_trees)

    def load(self, path: str, best_n_trees: Optional[int] = None) -> "LightGBMAdapter":
        self._booster = self._lgb.Booster(model_file=path)
        self.best_n_trees = best_n_trees or self._booster.current_iteration()
        self._fitted = True
        return self
