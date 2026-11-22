"""
pipeline.py — The end-to-end TeasCo-FT tree-adapter run
=======================================================

One call, :func:`run_experiment`, takes a training set and any number of test
sets through the whole cascade:

1. **Features** — MiniLM embedding (384) + the local Student SLM's confidence
   (1) for every training and test text, cached by content.
2. **Teacher** — the 70B Teacher labels the *training* texts once, offline,
   with a per-text soft-label cache so a re-run costs nothing.
3. **Adapters** — each selected tree adapter is trained under each selected
   supervision signal:

   ``ground_truth``   the dataset's own labels (needs a label column)
   ``teacher_hard``   the Teacher's argmax pseudo-labels
   ``teacher_soft``   the Teacher's full distributions — knowledge distillation
                      proper, via row expansion (see :func:`expand_soft_labels`)

4. **Evaluation** — macro/weighted F1 and accuracy per test set, next to the
   majority-class baseline and the frozen Student's own zero-shot score.

Collapse guards
---------------
The failure this pipeline actually has is silent: a model that predicts one
class for everything and posts a plausible 0.1–0.4 macro F1. Three guards run
every time and land in :attr:`ExperimentResult.warnings`:

1. the Student's confidence column must not be constant (a dead feature);
2. the Teacher must not give every text the same pseudo-label — teacher
   supervisions are dropped (or the run fails, if there is nothing else);
3. no trained adapter may emit a single class on a test set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from teasco_tree.adapters import DEFAULT_GATE_MODE, get_adapter, resolve_name
from teasco_tree.data import Dataset
from teasco_tree.features import FeatureBuilder, HashingEmbedder, SentenceEmbedder
from teasco_tree.llm import MockStudentLLM, MockTeacher, OllamaLLM, OpenAITeacher
from teasco_tree.llm.teacher import TeacherFailureError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
RUNS_DIR = PROJECT_ROOT / "runs"

SUPERVISIONS = ("ground_truth", "teacher_hard", "teacher_soft")
SUPERVISION_LABEL = {
    "ground_truth": "Ground truth",
    "teacher_hard": "Teacher pseudo-labels",
    "teacher_soft": "Teacher soft labels (KD)",
}
_MODEL_EXT = {"xgboost": ".json", "catboost": ".cbm", "lightgbm": ".txt"}

#: progress(stage, fraction in [0, 1], message)
ProgressFn = Callable[[str, float, str], None]


def _noop_progress(stage: str, fraction: float, message: str) -> None:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StudentConfig:
    """The frozen Student SLM that supplies feature column 384."""

    backend: str = "ollama"              # "ollama" | "mock"
    model: str = "qwen2.5:0.5b"
    host: str = "http://localhost:11434"
    scoring: str = "logprobs"            # "logprobs" | "json"
    debias: bool = True


@dataclass
class TeacherConfig:
    """The offline 70B Teacher. ``api_key`` is never written to disk."""

    backend: str = "api"                 # "api" | "mock" | "none"
    model: str = "meta-llama/llama-3.3-70b-instruct"
    base_url: Optional[str] = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = field(default=None, repr=False)
    extra_body: Optional[Dict[str, Any]] = None
    batch_size: int = 25
    max_failure_rate: float = 0.05
    tokens_per_minute: Optional[int] = None

    def public(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("api_key", None)
        return data


@dataclass
class TrainConfig:
    """Adapters, supervision signals and boosting hyper-parameters."""

    adapters: List[str] = field(default_factory=lambda: ["xgboost", "catboost", "lightgbm"])
    supervisions: List[str] = field(default_factory=lambda: list(SUPERVISIONS))
    gate_modes: Dict[str, str] = field(default_factory=dict)
    embedder: str = "minilm"             # "minilm" | "hashing"
    class_weight: Optional[str] = "balanced"
    n_estimators: int = 400
    learning_rate: float = 0.1
    max_depth: int = 6
    rounds_per_stage: int = 25
    patience: int = 6
    gate_margin: float = 0.10
    #: Weight kept by already-correct rows in staged gating. 0.0 is the strict
    #: reading of the gated rule, and it overshoots: a stage retrained on only
    #: the current mistakes flips the ones it had right, and validation F1
    #: swings by up to 0.68 between stages (scripts/simulate_gate_modes.py).
    #: 0.1 keeps the same scores with none of the swing.
    gate_floor: float = 0.10
    val_split: float = 0.15
    random_state: int = 42

    def gate_mode_for(self, adapter: str) -> str:
        return self.gate_modes.get(adapter, DEFAULT_GATE_MODE[adapter])


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def make_student(cfg: StudentConfig, classes: Sequence[str],
                 class_descriptions: Optional[Dict[str, str]] = None):
    if cfg.backend == "mock":
        return MockStudentLLM(classes)
    if cfg.backend != "ollama":
        raise ValueError(f"Unknown student backend {cfg.backend!r}")
    student = OllamaLLM(
        classes=classes, model_name=cfg.model, host=cfg.host,
        scoring=cfg.scoring, debias=cfg.debias,
        class_descriptions=class_descriptions,
        # Class order and wording change the prompt, so they belong in the key.
        cache_path=CACHE_DIR / "student" / (
            f"{_slug(cfg.model)}_{_digest(list(classes), class_descriptions)}.json"
        ),
    )
    if not student.is_available():
        raise RuntimeError(
            f"Student model {cfg.model!r} is not being served at {cfg.host}. "
            f"Start Ollama and run `ollama pull {cfg.model}`."
        )
    return student


def make_teacher(cfg: TeacherConfig, classes: Sequence[str],
                 class_descriptions: Optional[Dict[str, str]] = None):
    if cfg.backend == "mock":
        return MockTeacher(classes)
    if cfg.backend != "api":
        raise ValueError(f"Unknown teacher backend {cfg.backend!r}")
    return OpenAITeacher(
        api_key=cfg.api_key or "not-needed-for-local-servers",
        model=cfg.model,
        classes=list(classes),
        base_url=cfg.base_url,
        batch_size=cfg.batch_size,
        max_failure_rate=cfg.max_failure_rate,
        tokens_per_minute=cfg.tokens_per_minute,
        extra_body=cfg.extra_body,
        class_descriptions=class_descriptions,
    )


def make_embedder(name: str):
    if name == "hashing":
        return HashingEmbedder()
    if name == "minilm":
        return SentenceEmbedder()
    raise ValueError(f"Unknown embedder {name!r}")


def _digest(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


class SoftLabelCache:
    """
    Per-text Teacher soft labels on disk, keyed by model + classes + wording.

    Failed texts are never cached, so the next run retries them instead of
    replaying a uniform tie forever.
    """

    def __init__(self, model: str, classes: Sequence[str],
                 class_descriptions: Optional[Dict[str, str]] = None,
                 cache_dir: Path = CACHE_DIR / "teacher"):
        self.path = Path(cache_dir) / (
            f"{_slug(model)}_{_digest(list(classes), class_descriptions or {})}.json"
        )
        self._data: Dict[str, Dict[str, float]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                logger.warning("Teacher cache %s is corrupt; starting empty.", self.path)

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()

    def get(self, text: str) -> Optional[Dict[str, float]]:
        return self._data.get(self.key(text))

    def put(self, text: str, dist: Dict[str, float]) -> None:
        self._data[self.key(text)] = dist

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data))


def label_with_teacher(
    teacher, texts: List[str], classes: Sequence[str], cache: Optional[SoftLabelCache],
    progress: ProgressFn = _noop_progress,
) -> Tuple[List[Dict[str, float]], List[int], Dict[str, int]]:
    """Soft labels for every text, calling the Teacher only for uncached ones.

    Returns ``(soft_labels, failed_indices, usage)``; failed rows carry a
    uniform distribution and must be dropped before training.
    """
    soft: List[Optional[Dict[str, float]]] = [None] * len(texts)
    todo: List[int] = []
    for i, text in enumerate(texts):
        hit = cache.get(text) if cache is not None else None
        if hit is not None and set(hit) == set(classes):
            soft[i] = hit
        else:
            todo.append(i)

    failed: List[int] = []
    usage: Dict[str, int] = {"calls": 0, "total_tokens": 0, "cached": len(texts) - len(todo)}
    if todo:
        progress("teacher", 0.0, f"Teacher labeling {len(todo)} uncached texts "
                                 f"({usage['cached']} cached)")
        todo_texts = [texts[i] for i in todo]
        try:
            fresh = teacher.generate_soft_labels(
                todo_texts,
                progress_callback=lambda done, total: progress(
                    "teacher", done / total, f"Teacher labeled {done}/{total}"),
            )
        finally:
            usage.update({k: v for k, v in getattr(teacher, "usage", {}).items()
                          if k in ("calls", "total_tokens", "rate_limit_hits")})
        failed_local = set(teacher.failed_indices)
        for j, i in enumerate(todo):
            soft[i] = fresh[j]
            if j in failed_local:
                failed.append(i)
            elif cache is not None:
                cache.put(texts[i], fresh[j])
        if cache is not None:
            cache.save()
    progress("teacher", 1.0, "Teacher labels ready")
    return [s for s in soft], sorted(failed), usage  # type: ignore[misc]


def expand_soft_labels(
    X: np.ndarray, soft: np.ndarray, min_weight: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft-label distillation for a model that only accepts hard labels.

    Each row becomes one copy per class, labelled with that class and weighted
    by the Teacher's probability for it. Weighted cross-entropy over the
    expanded set equals cross-entropy against the soft distribution, so the
    boosted trees are fitted to the Teacher's full belief — including how sure
    it was — rather than to its argmax. (The same trick v1 TeasCo-FT uses with
    logistic regression.) Near-zero copies are dropped; they contribute nothing.

    Returns ``(X_expanded, y_expanded, weights, source_row)``; the gate needs
    ``source_row`` to judge each copy by its text's label, not its own.
    """
    n, k = soft.shape
    rows = np.repeat(np.arange(n), k)
    labels = np.tile(np.arange(k), n)
    weights = soft.reshape(-1)
    keep = weights >= min_weight
    return X[rows[keep]], labels[keep], weights[keep], rows[keep]


def majority_baseline_f1(labels: Sequence[str]) -> float:
    majority = Counter(labels).most_common(1)[0][0]
    return float(f1_score(list(labels), [majority] * len(labels),
                          average="macro", zero_division=0))


def _metrics(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> Dict[str, Any]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(classes), average="weighted", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(classes)).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    classes: List[str]
    config: Dict[str, Any]
    train_size: int
    fit_summaries: List[Dict[str, Any]]
    eval_rows: List[Dict[str, Any]]
    reference_rows: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]
    warnings: List[str]
    predictions: Dict[str, List[Dict[str, Any]]]
    timings: Dict[str, float]
    models: Dict[str, Any] = field(default_factory=dict, repr=False)
    feature_builder: Any = field(default=None, repr=False)
    run_dir: Optional[Path] = None

    def payload(self) -> Dict[str, Any]:
        """JSON-serialisable summary (no models, no API key)."""
        return {
            "classes": self.classes,
            "config": self.config,
            "train_size": self.train_size,
            "fit_summaries": self.fit_summaries,
            "results": self.eval_rows,
            "reference": self.reference_rows,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "timings": self.timings,
        }

    def predict(self, texts: List[str], model_key: str) -> Tuple[List[str], np.ndarray]:
        adapter = self.models[model_key]
        X = self.feature_builder.build(list(texts))
        proba = adapter.predict_proba(X)
        return [self.classes[i] for i in proba.argmax(axis=1)], proba


def model_key(adapter: str, supervision: str) -> str:
    return f"{adapter}__{supervision}"


# ─────────────────────────────────────────────────────────────────────────────
# The run
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    train: Dataset,
    tests: Sequence[Dataset],
    classes: Optional[Sequence[str]] = None,
    class_descriptions: Optional[Dict[str, str]] = None,
    student_cfg: Optional[StudentConfig] = None,
    teacher_cfg: Optional[TeacherConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    progress: ProgressFn = _noop_progress,
    run_dir: Optional[Path] = None,
    embedder=None,
    student=None,
    teacher=None,
) -> ExperimentResult:
    """
    Train every selected tree adapter under every selected supervision signal
    and evaluate on every test set. ``embedder``/``student``/``teacher`` may be
    passed pre-built (the UI caches the MiniLM model across runs).
    """
    student_cfg = student_cfg or StudentConfig()
    teacher_cfg = teacher_cfg or TeacherConfig()
    train_cfg = train_cfg or TrainConfig()
    timings: Dict[str, float] = {}
    warnings: List[str] = []
    t_run = time.time()

    # ── Classes & supervision plan ─────────────────────────────────────────
    if classes is None:
        if not train.has_labels:
            raise ValueError("Unlabelled training data needs an explicit class list.")
        classes = sorted(set(train.labels))
    classes = list(classes)
    if len(classes) < 2:
        raise ValueError(f"Need at least two classes, got {classes}.")
    class_descriptions = {c: (class_descriptions or {}).get(c, c) for c in classes}

    if train.has_labels:
        unknown = sorted(set(train.labels) - set(classes))
        if unknown:
            raise ValueError(f"Training labels {unknown} are not in classes {classes}.")

    adapters = [resolve_name(a) for a in train_cfg.adapters]
    if not adapters:
        raise ValueError("Select at least one tree adapter.")
    supervisions = [s for s in train_cfg.supervisions if s in SUPERVISIONS]
    if "ground_truth" in supervisions and not train.has_labels:
        supervisions.remove("ground_truth")
        warnings.append("Training data has no labels, so 'ground_truth' was skipped — "
                        "the adapters learn from the Teacher alone.")
    if teacher_cfg.backend == "none":
        dropped = [s for s in supervisions if s.startswith("teacher")]
        supervisions = [s for s in supervisions if not s.startswith("teacher")]
        if dropped:
            warnings.append(f"No Teacher configured; skipped {dropped}.")
    if not supervisions:
        raise ValueError("Nothing to train: no usable supervision signal "
                         "(unlabelled data needs a Teacher).")
    needs_teacher = any(s.startswith("teacher") for s in supervisions)

    # ── Test sets that can be scored ───────────────────────────────────────
    scored_tests: List[Dataset] = []
    for ds in tests:
        if ds.has_labels:
            extra = sorted(set(ds.labels) - set(classes))
            if extra:
                warnings.append(f"Test set '{ds.name}' has labels {extra} outside {classes}; "
                                f"it gets predictions only, no scores.")
        scored_tests.append(ds)

    # ── 1 · Features ───────────────────────────────────────────────────────
    t0 = time.time()
    progress("setup", 0.0, "Loading the embedder and Student SLM")
    embedder = embedder or make_embedder(train_cfg.embedder)
    student = student or make_student(student_cfg, classes, class_descriptions)
    builder = FeatureBuilder(embedder=embedder, llm=student,
                             cache_dir=str(CACHE_DIR / "features"), verbose=False)

    def build(ds: Dataset, stage: str) -> np.ndarray:
        progress(stage, 0.0, f"Student SLM + embeddings for '{ds.name}' ({len(ds)} texts)")
        X = builder.build(ds.texts, progress_callback=lambda d, t: progress(
            stage, d / t, f"Student SLM on '{ds.name}': {d}/{t}"))
        progress(stage, 1.0, f"Features ready for '{ds.name}'")
        return X

    X_train = build(train, "features_train")
    X_tests = {ds.name: build(ds, "features_test") for ds in scored_tests}
    timings["features"] = time.time() - t0

    # Guard 1 — a constant confidence column is a dead feature.
    conf = X_train[:, -1]
    diagnostics: Dict[str, Any] = {
        "student_distinct_confidences": int(len(np.unique(np.round(conf, 4)))),
        "student_confidence_mean": float(conf.mean()),
        "student_confidence_std": float(conf.std()),
    }
    if diagnostics["student_distinct_confidences"] <= 1:
        warnings.append("Guard 1: the Student's confidence is identical for every training "
                        "text, so feature column 384 carries no information. Use "
                        "scoring='logprobs' or a different Student model.")

    # The Student's own zero-shot answers, as a reference row (cached per text).
    student_preds: Dict[str, List[str]] = {}
    for ds in [train] + scored_tests:
        student_preds[ds.name], _ = student.predict(ds.texts)
    if train.has_labels:
        s_train = student_preds[train.name]
        correct = np.asarray(s_train) == np.asarray(train.labels)
        diagnostics["student_train_macro_f1"] = _metrics(train.labels, s_train, classes)["macro_f1"]
        if conf.std() > 0 and 0 < correct.mean() < 1:
            diagnostics["student_conf_correct_corr"] = float(np.corrcoef(conf, correct)[0, 1])

    # ── 2 · Teacher ────────────────────────────────────────────────────────
    soft_matrix: Optional[np.ndarray] = None
    failed: List[int] = []
    if needs_teacher:
        t0 = time.time()
        teacher = teacher or make_teacher(teacher_cfg, classes, class_descriptions)
        cache = None if teacher_cfg.backend == "mock" else SoftLabelCache(
            teacher_cfg.model, classes, class_descriptions)
        soft, failed, usage = label_with_teacher(teacher, train.texts, classes, cache, progress)
        soft_matrix = np.array([[d[c] for c in classes] for d in soft], dtype=np.float64)
        timings["teacher"] = time.time() - t0
        diagnostics["teacher_usage"] = usage
        diagnostics["teacher_failed"] = len(failed)
        if failed:
            warnings.append(f"The Teacher could not label {len(failed)} training text(s); "
                            f"they were dropped rather than trained on as uniform ties.")

    keep = np.array([i for i in range(len(train)) if i not in set(failed)])
    if len(keep) < 10:
        raise ValueError(f"Only {len(keep)} usable training rows.")

    teacher_labels: Optional[np.ndarray] = None
    if soft_matrix is not None:
        teacher_labels = soft_matrix.argmax(axis=1)
        dist = Counter(teacher_labels[keep].tolist())
        diagnostics["teacher_label_distribution"] = {classes[k]: v for k, v in dist.items()}
        if train.has_labels:
            gold_idx = np.array([classes.index(l) for l in train.labels])
            m = _metrics([classes[i] for i in gold_idx[keep]],
                         [classes[i] for i in teacher_labels[keep]], classes)
            diagnostics["teacher_train_agreement"] = {k: m[k] for k in ("macro_f1", "accuracy")}
        # Guard 2 — a single pseudo-label distils into a constant model.
        if len(dist) == 1:
            msg = (f"Guard 2: the Teacher gave all {len(keep)} usable texts the same label "
                   f"({classes[next(iter(dist))]}). Teacher supervisions were skipped.")
            if "ground_truth" in supervisions:
                warnings.append(msg)
                supervisions = ["ground_truth"]
            else:
                raise TeacherFailureError(msg)

    # ── 3 · Train ──────────────────────────────────────────────────────────
    t0 = time.time()
    y_gold = (np.array([classes.index(l) for l in train.labels]) if train.has_labels else None)
    stratify_on = (y_gold if y_gold is not None else teacher_labels)[keep]
    counts = np.bincount(stratify_on, minlength=len(classes))
    tr_idx, val_idx = train_test_split(
        keep, test_size=train_cfg.val_split, random_state=train_cfg.random_state,
        stratify=stratify_on if counts[counts > 0].min() >= 2 else None,
    )

    fit_summaries: List[Dict[str, Any]] = []
    models: Dict[str, Any] = {}
    plan = [(a, s) for s in supervisions for a in adapters]
    for step, (adapter_name, supervision) in enumerate(plan):
        key = model_key(adapter_name, supervision)
        progress("train", step / len(plan), f"Training {adapter_name} on {SUPERVISION_LABEL[supervision]}")
        adapter = get_adapter(
            adapter_name, classes=classes,
            gate_mode=train_cfg.gate_mode_for(adapter_name),
            n_estimators=train_cfg.n_estimators, learning_rate=train_cfg.learning_rate,
            max_depth=train_cfg.max_depth, rounds_per_stage=train_cfg.rounds_per_stage,
            patience=train_cfg.patience, gate_margin=train_cfg.gate_margin,
            gate_floor=train_cfg.gate_floor,
            class_weight=train_cfg.class_weight, random_state=train_cfg.random_state,
            verbose=False,
        )
        weights = gate_labels = None
        if supervision == "ground_truth":
            X_fit, y_fit, y_val = X_train[tr_idx], y_gold[tr_idx], y_gold[val_idx]
        elif supervision == "teacher_hard":
            X_fit, y_fit, y_val = X_train[tr_idx], teacher_labels[tr_idx], teacher_labels[val_idx]
        else:  # teacher_soft
            X_fit, y_fit, weights, source = expand_soft_labels(X_train[tr_idx], soft_matrix[tr_idx])
            gate_labels = teacher_labels[tr_idx][source]
            y_val = teacher_labels[val_idx]
        # Teacher variants validate against Teacher labels too: the gold labels
        # must not leak into a run that claims to learn from the Teacher alone.

        t_fit = time.time()
        summary = adapter.fit(X_fit, y_fit, X_train[val_idx], y_val,
                              sample_weight=weights, gate_labels=gate_labels)
        fit_summaries.append({
            "key": key, "adapter": adapter_name, "supervision": supervision,
            "gate_mode": adapter.gate_mode, "fit_seconds": time.time() - t_fit,
            "best_val_f1": summary["best_val_f1"], "best_n_trees": summary["best_n_trees"],
            "train_rows": int(len(y_fit)), "history": summary["history"],
        })
        models[key] = adapter
    progress("train", 1.0, f"Trained {len(plan)} adapter(s)")
    timings["train"] = time.time() - t0

    # ── 4 · Evaluate ───────────────────────────────────────────────────────
    t0 = time.time()
    eval_rows: List[Dict[str, Any]] = []
    reference_rows: List[Dict[str, Any]] = []
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    for ds in scored_tests:
        X = X_tests[ds.name]
        scorable = ds.has_labels and set(ds.labels) <= set(classes)
        rows = [{"text": t} for t in ds.texts]
        if ds.has_labels:
            for r, l in zip(rows, ds.labels):
                r["label"] = l
        for r, p in zip(rows, student_preds[ds.name]):
            r["student_slm"] = p

        if scorable:
            baseline = majority_baseline_f1(ds.labels)
            reference_rows.append({"test_set": ds.name, "kind": "majority_baseline",
                                   "macro_f1": baseline, "n": len(ds)})
            sm = _metrics(ds.labels, student_preds[ds.name], classes)
            reference_rows.append({"test_set": ds.name, "kind": "student_zero_shot",
                                   "macro_f1": sm["macro_f1"], "accuracy": sm["accuracy"], "n": len(ds)})

        for key, adapter in models.items():
            proba = adapter.predict_proba(X)
            preds = [classes[i] for i in proba.argmax(axis=1)]
            for r, p, pr in zip(rows, preds, proba.max(axis=1)):
                r[key] = p
                r[f"{key}__confidence"] = round(float(pr), 4)
            if not scorable:
                continue
            m = _metrics(ds.labels, preds, classes)
            predicted = Counter(preds)
            adapter_name, supervision = key.split("__")
            eval_rows.append({
                "key": key, "adapter": adapter_name, "supervision": supervision,
                "test_set": ds.name, "n": len(ds), **m,
                "baseline_macro_f1": baseline,
                "student_macro_f1": sm["macro_f1"],
                "predicted_distribution": dict(predicted),
                "collapsed": len(predicted) == 1,
            })
            # Guard 3 — one class for everything is a dead model, whatever its score.
            if len(predicted) == 1:
                warnings.append(f"Guard 3: {adapter_name} / {SUPERVISION_LABEL[supervision]} "
                                f"predicts only '{next(iter(predicted))}' on '{ds.name}'.")
        predictions[ds.name] = rows
    timings["evaluate"] = time.time() - t0
    timings["total"] = time.time() - t_run

    config = {
        "student": asdict(student_cfg),
        "teacher": teacher_cfg.public(),
        "train": asdict(train_cfg),
        "class_descriptions": class_descriptions,
        "train_dataset": train.name,
        "test_sets": {ds.name: len(ds) for ds in scored_tests},
        "embedder": getattr(embedder, "model_name", type(embedder).__name__),
    }
    result = ExperimentResult(
        classes=classes, config=config, train_size=int(len(keep)),
        fit_summaries=fit_summaries, eval_rows=eval_rows, reference_rows=reference_rows,
        diagnostics=diagnostics, warnings=warnings, predictions=predictions,
        timings=timings, models=models, feature_builder=builder,
    )
    if run_dir is not None:
        save_run(result, run_dir)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def new_run_dir(label: str = "run") -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{stamp}_{_slug(label)[:40]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(result: ExperimentResult, run_dir: Path) -> Path:
    """Write results.json, per-test-set prediction CSVs and every trained model."""
    import pandas as pd

    run_dir = Path(run_dir)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)

    (run_dir / "results.json").write_text(json.dumps(result.payload(), indent=2, default=str))
    for name, rows in result.predictions.items():
        pd.DataFrame(rows).to_csv(run_dir / "predictions" / f"{_slug(name)}.csv", index=False)

    manifest = {}
    for key, adapter in result.models.items():
        adapter_name = key.split("__")[0]
        filename = f"{key}{_MODEL_EXT[adapter_name]}"
        adapter.save(str(run_dir / "models" / filename))
        manifest[key] = {"file": filename, "adapter": adapter_name,
                         "gate_mode": adapter.gate_mode, "best_n_trees": int(adapter.best_n_trees)}
    (run_dir / "models" / "manifest.json").write_text(json.dumps({
        "classes": result.classes,
        "student": result.config["student"],
        "embedder": result.config["embedder"],
        "class_descriptions": result.config["class_descriptions"],
        "models": manifest,
    }, indent=2))
    result.run_dir = run_dir
    return run_dir


class SavedCascade:
    """A trained run reloaded from disk, ready to classify new texts.

    Rebuilds the same 385-dim features — so the same Student SLM must be
    available — and applies the stored adapter at its early-stopped size.
    """

    def __init__(self, run_dir: Path, embedder=None, student=None):
        self.run_dir = Path(run_dir)
        manifest = json.loads((self.run_dir / "models" / "manifest.json").read_text())
        self.classes: List[str] = manifest["classes"]
        self.model_info: Dict[str, Dict[str, Any]] = manifest["models"]
        student_cfg = StudentConfig(**manifest["student"])
        if embedder is None:
            embedder = make_embedder("hashing" if manifest["embedder"] == HashingEmbedder.model_name
                                     else "minilm")
        if student is None:
            student = make_student(student_cfg, self.classes, manifest.get("class_descriptions"))
        self.builder = FeatureBuilder(embedder=embedder, llm=student,
                                      cache_dir=str(CACHE_DIR / "features"), verbose=False)
        self._adapters: Dict[str, Any] = {}

    @property
    def model_keys(self) -> List[str]:
        return list(self.model_info)

    def adapter(self, key: str):
        if key not in self._adapters:
            info = self.model_info[key]
            adapter = get_adapter(info["adapter"], classes=self.classes,
                                  gate_mode=info["gate_mode"], verbose=False)
            adapter.load(str(self.run_dir / "models" / info["file"]),
                         best_n_trees=info["best_n_trees"])
            self._adapters[key] = adapter
        return self._adapters[key]

    def predict(self, texts: List[str], key: str, progress_callback=None) -> Tuple[List[str], np.ndarray]:
        X = self.builder.build(list(texts), progress_callback=progress_callback)
        proba = self.adapter(key).predict_proba(X)
        return [self.classes[i] for i in proba.argmax(axis=1)], proba


def list_runs() -> List[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted((p for p in RUNS_DIR.iterdir() if (p / "models" / "manifest.json").exists()),
                  reverse=True)
