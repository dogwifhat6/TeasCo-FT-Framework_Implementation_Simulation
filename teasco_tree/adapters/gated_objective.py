"""
gated_objective.py — The Gated Update Rule for Gradient-Boosted Trees
======================================================================

Porting :class:`GatedLoss (the MLP loss in the original teascoFT_v2)` from a neural adapter to a
tree ensemble is the central technical problem in swapping the MLP out for
XGBoost/CatBoost/LightGBM.

The MLP realises the rule by zeroing the *loss* of correctly-classified samples,
so no gradient reaches the weights. A boosted ensemble has no weights to
backprop into — each round fits a new tree to the gradient/hessian of the loss
with respect to the current ensemble margin. The faithful translation is
therefore to zero those **per-sample gradients directly**:

    grad_i = 0,  hess_i = eps      if argmax(p_i) == y_i   (already correct)
    grad_i = p_i - Y_i,            otherwise               (standard softmax CE)
    hess_i = 2 * p_i * (1 - p_i)

A tree fitted against those statistics can only reduce loss on samples the
ensemble currently gets wrong: leaf values are ``-sum(grad)/(sum(hess)+lambda)``,
which collapses to ~0 wherever the gate is closed. Because boosting recomputes
the margin every round, the gate is re-evaluated every round — the same dynamic
re-gating the MLP performs every mini-batch.

Two details matter in practice:

* ``eps`` must be a small *positive* hessian rather than exactly 0. A zero
  hessian makes the leaf-weight denominator degenerate and, since XGBoost's
  ``min_child_weight`` is a sum of hessians, it also keeps gated samples from
  propping open splits they contribute no gradient to.
* Label smoothing is applied to the one-hot target to mirror the MLP's
  ``CrossEntropyLoss(label_smoothing=0.05)``, keeping the two families
  comparable rather than differing by an incidental regularisation choice.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def softmax(margins: np.ndarray) -> np.ndarray:
    """
    Numerically stable row-wise softmax.

    Parameters
    ----------
    margins : np.ndarray, shape (N, K)
        Raw ensemble margins.

    Returns
    -------
    np.ndarray, shape (N, K) — probabilities summing to 1 per row.
    """
    shifted = margins - margins.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def gated_softmax_grad_hess(
    margins: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    gate: bool = True,
    label_smoothing: float = 0.0,
    eps_hess: float = 1e-6,
    sample_weight: Optional[np.ndarray] = None,
    gate_margin: float = 0.0,
    gate_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient and hessian of the (optionally gated) softmax cross-entropy.

    Parameters
    ----------
    margins : np.ndarray, shape (N, K)
        Current raw ensemble margins.
    y : np.ndarray, shape (N,)
        Integer targets.
    num_classes : int
        Number of classes K.
    gate : bool
        Apply the gated update rule (zero the signal on correct samples).
        Set False for an ungated control run.
    label_smoothing : float
        Mirrors ``CrossEntropyLoss(label_smoothing=...)`` used by the MLP.
    eps_hess : float
        Small positive hessian assigned to gated-out samples.
    sample_weight : np.ndarray, optional
        Per-sample weights (e.g. class balancing), broadcast over classes.
    gate_labels : np.ndarray, optional
        Labels the gate judges correctness against, when they differ from the
        training targets ``y`` (soft-label expansion). Defaults to ``y``.
    gate_margin : float
        Confidence band that keeps *correct but uncertain* samples alive. A
        sample is silenced only when it is correct **and** its top-vs-runner-up
        probability gap exceeds this value.

        This matters more than it looks. A strict correctness gate
        (``gate_margin=0``) is degenerate for boosting: the first round predicts
        from a constant margin, so every sample ties to class 0, the gate
        silences that entire class, and every surviving sample pushes the same
        direction. With no contrast between samples there is no split to find,
        and XGBoost emits single-leaf trees forever — a constant model. The MLP
        escapes this only incidentally, because random initialisation scatters
        its first predictions across both classes.

        A positive margin restores the contrast by keeping near-boundary
        samples in play. It is the same rule v1 already uses to mine hard
        examples in ``KnowledgeDistillationTrainer._identify_hard_examples``:
        a sample is hard when it is *wrong or low-confidence*.

    Returns
    -------
    (grad, hess) : both np.ndarray of shape (N, K)
    """
    proba = softmax(margins)

    target = np.zeros_like(proba)
    target[np.arange(len(y)), y] = 1.0
    if label_smoothing > 0.0:
        target = (
            target * (1.0 - label_smoothing) + label_smoothing / num_classes
        )

    grad = proba - target
    hess = 2.0 * proba * (1.0 - proba)  # XGBoost's softmax convention

    if gate:
        # Silence samples that are already correct *and* confidently so.
        # With soft-label expansion every text has one copy per class, so
        # "correct" must be judged against the text's own label (the Teacher's
        # argmax), not the copy's — otherwise the minority copy is always
        # "wrong", never silenced, and the gate drags the model toward it.
        correct = proba.argmax(axis=1) == (y if gate_labels is None else gate_labels)

        if gate_margin > 0.0:
            if num_classes > 1:
                partitioned = np.partition(proba, -2, axis=1)
                confidence_gap = partitioned[:, -1] - partitioned[:, -2]
            else:  # pragma: no cover - degenerate single-class case
                confidence_gap = np.ones(len(y))
            silenced = correct & (confidence_gap > gate_margin)
        else:
            silenced = correct

        grad[silenced] = 0.0
        hess[silenced] = eps_hess

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1, 1)
        grad = grad * w
        hess = hess * w
        # Keep the hessian strictly positive even at zero weight.
        hess = np.maximum(hess, eps_hess)

    return grad.astype(np.float32), hess.astype(np.float32)


def make_xgboost_gated_objective(
    num_classes: int,
    gate: bool = True,
    label_smoothing: float = 0.0,
    eps_hess: float = 1e-6,
    sample_weight: Optional[np.ndarray] = None,
    gate_margin: float = 0.0,
    warmup_rounds: int = 0,
    rounds_completed: int = 0,
    gate_labels: Optional[np.ndarray] = None,
) -> Callable:
    """
    Build a custom XGBoost objective implementing the gated update rule.

    The returned callable matches XGBoost's ``obj(preds, dtrain)`` signature and
    is passed to :func:`xgboost.train` via ``obj=``. It tolerates both the
    flattened ``(N*K,)`` and the modern ``(N, K)`` prediction layouts.

    Parameters
    ----------
    warmup_rounds : int
        Boosting rounds trained *ungated* before the gate engages. Boosting
        begins from a constant prediction, which a strict gate collapses (see
        :func:`gated_softmax_grad_hess`); a warmup lets the ensemble reach a
        non-degenerate state first, playing the role that random initialisation
        plays for the MLP.
    rounds_completed : int
        Rounds already trained in earlier stages, so warmup is counted over the
        whole ensemble rather than restarting each stage.

    Returns
    -------
    callable — suitable for ``xgb.train(..., obj=<this>)``.
    """
    state = {"round": rounds_completed}

    def objective(preds: np.ndarray, dtrain) -> Tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label().astype(np.int64)

        margins = np.asarray(preds, dtype=np.float64)
        if margins.ndim == 1:
            margins = margins.reshape(-1, num_classes)

        gate_now = gate and state["round"] >= warmup_rounds
        state["round"] += 1

        grad, hess = gated_softmax_grad_hess(
            margins,
            y,
            num_classes=num_classes,
            gate=gate_now,
            label_smoothing=label_smoothing,
            eps_hess=eps_hess,
            sample_weight=sample_weight,
            gate_margin=gate_margin,
            gate_labels=gate_labels,
        )

        # Return in whatever layout XGBoost handed us.
        if np.asarray(preds).ndim == 1:
            return grad.reshape(-1), hess.reshape(-1)
        return grad, hess

    return objective


def gate_sample_weights(
    predictions: np.ndarray,
    y: np.ndarray,
    gate_floor: float = 0.0,
    base_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Sample weights implementing the gated rule for *staged* boosting.

    Used by adapters whose library does not expose a custom multiclass
    objective (CatBoost, LightGBM): instead of masking gradients, the ensemble
    is warm-started and the next stage of trees is fitted with correct samples
    down-weighted to ``gate_floor``.

    Parameters
    ----------
    predictions : np.ndarray, shape (N,)
        Current ensemble predictions (integer class ids).
    y : np.ndarray, shape (N,)
        Integer targets.
    gate_floor : float
        Weight retained by already-correct samples. ``0.0`` is the strict
        reading of the spec; a small positive value trades strictness for
        stability on tiny error sets.
    base_weight : np.ndarray, optional
        Multiplied into the gate (e.g. class-balancing weights).

    Returns
    -------
    (weights, n_incorrect)
    """
    wrong = predictions != y
    n_incorrect = int(wrong.sum())

    weights = np.where(wrong, 1.0, gate_floor).astype(np.float64)
    if base_weight is not None:
        weights = weights * np.asarray(base_weight, dtype=np.float64)

    return weights, n_incorrect
