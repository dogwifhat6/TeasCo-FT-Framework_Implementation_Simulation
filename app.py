"""
TeasCo-FT · Tree Adapters — Streamlit front end.

    streamlit run app.py

Upload a training set, pick test sets, train XGBoost / CatBoost / LightGBM
adapters on top of a local Student SLM with a 70B Teacher, and read the results
as tables, a macro-F1 heatmap, bar charts, training curves and confusion
matrices. Trained runs are saved under runs/ and can classify new text.
"""

# Must be the first import: it loads the boosting libraries' OpenMP runtime
# before sentence-transformers pulls in torch (see teasco_tree/_compat.py).
import teasco_tree  # noqa: F401  isort:skip

import io
import json
import math
import os
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from teasco_tree import plots
from teasco_tree.adapters import DEFAULT_GATE_MODE, DISPLAY_NAME, GATE_MODES, available_adapters
from teasco_tree.data import (
    SAMPLE_DATASETS, Dataset, clean_dataset, guess_columns, load_sample, read_table, to_dataset,
)
from teasco_tree.llm import TEACHER_PRESETS, MockStudentLLM
from teasco_tree.pipeline import (
    SUPERVISION_LABEL, SUPERVISIONS, SavedCascade, StudentConfig, TeacherConfig, TrainConfig,
    list_runs, make_embedder, make_teacher, new_run_dir, run_experiment, save_run,
)
from teasco_tree.system import load_env, ollama_models, total_memory_gb

ROOT = Path(__file__).resolve().parent
load_env(ROOT / ".env")

st.set_page_config(page_title="TeasCo-FT · Tree Adapters", page_icon="🌲", layout="wide")

NO_LABEL = "— none: let the Teacher label it —"
KNOWN_DESCRIPTIONS = {
    "sec": "security-related", "nonsec": "not security-related",
    "FR": "a functional requirement", "NFR": "a non-functional requirement",
    "PE": "a performance requirement", "US": "a usability requirement",
    "SE": "a security requirement", "O": "an operational requirement",
    "LF": "a look-and-feel requirement", "A": "an availability requirement",
    "MN": "a maintainability requirement", "SC": "a scalability requirement",
    "L": "a legal requirement", "FT": "a fault-tolerance requirement",
    "PO": "a portability requirement", "F": "a functional requirement",
}


@st.cache_resource(show_spinner="Loading the sentence embedder…")
def cached_embedder(name: str):
    return make_embedder(name)


@st.cache_resource(show_spinner=False)
def cached_saved_cascade(run_dir: str, demo: bool):
    manifest = json.loads((Path(run_dir) / "models" / "manifest.json").read_text())
    student = MockStudentLLM(manifest["classes"]) if demo else None
    embedder = cached_embedder("hashing" if manifest["embedder"] == "hashing-384" else "minilm")
    return SavedCascade(Path(run_dir), embedder=embedder, student=student)


def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=plots.SURFACE)
    plt.close(fig)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — models and training settings
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌲 TeasCo-FT")
    st.caption("Tree-adapter cascade · local Student SLM · 70B Teacher")

    mode = st.radio(
        "Run mode", ["Real models", "Offline demo"], horizontal=True,
        help="Offline demo swaps in keyword-based mock Student/Teacher models and a "
             "hashing embedder, so the whole flow runs in seconds with no Ollama, "
             "no API key and no downloads. Its numbers mean nothing.",
    )
    demo = mode == "Offline demo"
    config_errors: list = []

    # ── Student ────────────────────────────────────────────────────────────
    st.subheader("Student SLM · local")
    if demo:
        st.info("Mock Student (keyword-based).")
        student_cfg = StudentConfig(backend="mock", model="mock-student")
    else:
        host = st.text_input("Ollama host", "http://localhost:11434")
        local = ollama_models(host)
        if local is None:
            st.error("Ollama is not reachable. Start it (`ollama serve`) and pull a model.")
            config_errors.append("Ollama is not running.")
            student_model = st.text_input("Student model", "qwen2.5:0.5b")
        elif not local:
            st.error("Ollama has no models. Run `ollama pull qwen2.5:0.5b`.")
            config_errors.append("No local Ollama models.")
            student_model = "qwen2.5:0.5b"
        else:
            default = local.index("qwen2.5:0.5b") if "qwen2.5:0.5b" in local else 0
            student_model = st.selectbox("Student model (downloaded)", local, index=default)
        scoring = st.selectbox(
            "Confidence from", ["logprobs", "json"],
            format_func=lambda s: {"logprobs": "first-token logprobs (recommended)",
                                   "json": "self-reported JSON number"}[s],
            help="Small models asked to state a confidence answer ~1.0 to everything, which "
                 "makes the 385th feature constant. Logprobs give a real distribution.",
        )
        debias = st.checkbox("Ask in both option orders (debias)", value=True,
                             help="Cancels small models' preference for option A. Two calls per text.")
        student_cfg = StudentConfig(backend="ollama", model=student_model, host=host,
                                    scoring=scoring, debias=debias)

    # ── Teacher ────────────────────────────────────────────────────────────
    st.subheader("Teacher LLM · 70B")
    if demo:
        st.info("Mock Teacher (keyword-based soft labels).")
        teacher_cfg = TeacherConfig(backend="mock", model="mock-teacher", base_url=None)
    else:
        preset_key = st.selectbox("Teacher", list(TEACHER_PRESETS),
                                  format_func=lambda k: TEACHER_PRESETS[k]["label"])
        preset = TEACHER_PRESETS[preset_key]
        st.caption(preset["note"])
        memory = total_memory_gb()
        if preset["min_memory_gb"] and memory is not None and memory < preset["min_memory_gb"]:
            st.error(f"This machine has {memory:.0f} GB of memory; a local 70B model needs "
                     f"~{preset['min_memory_gb']} GB. Use the hosted preset instead.")
            config_errors.append("Not enough memory for a local 70B Teacher.")
        teacher_model = st.text_input("Model", preset["model"], key=f"tm_{preset_key}")
        base_url = st.text_input("Base URL", preset["base_url"], key=f"tb_{preset_key}")
        env_name = preset["api_key_env"]
        api_key = None
        if env_name:
            api_key = st.text_input(
                f"API key ({env_name})", os.environ.get(env_name, ""), type="password",
                key=f"tk_{preset_key}",
                help=f"Read from the {env_name} environment variable or .env; never saved.",
            ) or None
        batch_size = st.slider("Texts per Teacher call", 1, 50, 25,
                               help="The system prompt is paid once per call, so batching "
                                    "cuts tokens per text roughly 2–3x.")
        teacher_cfg = TeacherConfig(
            backend="api", model=teacher_model, base_url=base_url or None, api_key=api_key,
            extra_body=preset["extra_body"], batch_size=batch_size,
        )
        needs_key = env_name is not None and not api_key

    # ── Adapters ───────────────────────────────────────────────────────────
    st.subheader("Tree adapters")
    adapters = st.multiselect("Adapters", available_adapters(), default=available_adapters(),
                              format_func=lambda a: DISPLAY_NAME[a])
    supervisions = st.multiselect(
        "Train each adapter on", list(SUPERVISIONS), default=list(SUPERVISIONS),
        format_func=lambda s: SUPERVISION_LABEL[s],
        help="Ground truth uses your label column. Teacher pseudo-labels use the 70B "
             "model's top answer. Teacher soft labels train on its full probability "
             "distribution — knowledge distillation proper.",
    )
    with st.expander("Boosting settings"):
        gate_modes = {}
        for a in adapters:
            modes = GATE_MODES[a]
            gate_modes[a] = st.selectbox(
                f"{DISPLAY_NAME[a]} gate", modes, index=modes.index(DEFAULT_GATE_MODE[a]),
                key=f"gate_{a}",
                help="objective: zero the gradient of confidently-correct rows (XGBoost only). "
                     "staged: down-weight correct rows between blocks of trees. none: ungated.",
            )
        n_estimators = st.slider("Max trees", 50, 1000, 400, step=50)
        learning_rate = st.select_slider("Learning rate", [0.02, 0.05, 0.1, 0.2, 0.3], value=0.1)
        max_depth = st.slider("Max depth", 2, 10, 6)
        gate_floor = st.slider("Staged gate floor", 0.0, 1.0, 0.1, step=0.05,
                               help="Weight kept by already-correct rows. 0 is the strict rule "
                                    "and makes validation F1 swing between stages.")
        val_split = st.slider("Validation split (early stopping)", 0.05, 0.4, 0.15, step=0.05)
        seed = st.number_input("Random seed", value=42, step=1)
    train_cfg = TrainConfig(
        adapters=adapters, supervisions=supervisions, gate_modes=gate_modes,
        embedder="hashing" if demo else "minilm", n_estimators=n_estimators,
        learning_rate=learning_rate, max_depth=max_depth, gate_floor=gate_floor,
        val_split=val_split, random_state=int(seed),
    )


st.title("TeasCo-FT · Tree-adapter cascade")
st.caption(
    "A 70B Teacher labels your training data once, offline. XGBoost, CatBoost and "
    "LightGBM adapters learn from those labels on top of MiniLM embeddings plus a local "
    "Student SLM's confidence. At inference only the small local models run."
)
tab_data, tab_train, tab_results, tab_predict, tab_about = st.tabs(
    ["📂 Data", "🚀 Train", "📊 Results", "🔮 Predict", "ℹ️ How it works"])


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def column_picker(df: pd.DataFrame, key: str, allow_no_label: bool):
    text_guess, label_guess = guess_columns(df)
    cols = list(df.columns)
    c1, c2 = st.columns(2)
    text_col = c1.selectbox("Text column", cols, index=cols.index(text_guess), key=f"{key}_text")
    label_options = ([NO_LABEL] if allow_no_label else ["— no labels (predict only) —"]) + \
        [c for c in cols if c != text_col]
    label_index = label_options.index(label_guess) if label_guess in label_options else 0
    label_col = c2.selectbox("Label column", label_options, index=label_index, key=f"{key}_label")
    return text_col, (None if label_col.startswith("—") else label_col)


data_errors: list = []
train_ds = None
tests: list = []
classes = None
descriptions: dict = {}

with tab_data:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Training data")
        source = st.radio("Source", ["Upload a file", "Sample dataset"], horizontal=True,
                          index=1, key="train_source")
        train_df = None
        train_name = None
        if source == "Upload a file":
            upload = st.file_uploader("CSV, TSV, semicolon-delimited, Excel or JSON",
                                      type=["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl"],
                                      key="train_upload")
            if upload is not None:
                try:
                    raw = read_table(upload)
                    st.caption(f"{len(raw)} rows · columns: {', '.join(map(str, raw.columns))}")
                    text_col, label_col = column_picker(raw, "train", allow_no_label=True)
                    train_df = clean_dataset(raw, text_col, label_col)
                    train_name = Path(upload.name).stem
                except Exception as exc:
                    st.error(f"Could not read the file: {exc}")
        else:
            sample = st.selectbox("Sample", list(SAMPLE_DATASETS), key="train_sample")
            st.caption(SAMPLE_DATASETS[sample]["about"])
            train_df = load_sample(sample)
            train_name = sample.split(" (")[0]

        if train_df is not None:
            if len(train_df) < 30:
                data_errors.append("The training set needs at least 30 usable rows.")
            has_labels = "label" in train_df.columns
            if has_labels:
                counts = train_df["label"].value_counts()
                classes = sorted(counts.index)
                st.write(f"**{len(train_df)} rows** after removing blanks and duplicates · "
                         f"{len(classes)} classes")
                st.dataframe(counts.rename("rows").to_frame().T, hide_index=True)
                if len(classes) < 2:
                    data_errors.append("The label column has only one class.")
                if len(classes) > 26:
                    data_errors.append("At most 26 classes are supported (one letter each in "
                                       "the Student's prompt).")
                rare = counts[counts < 2]
                if not rare.empty:
                    data_errors.append(f"Classes with fewer than 2 rows: {rare.to_dict()}")
            else:
                st.write(f"**{len(train_df)} unlabelled rows** — the Teacher will label them.")
                typed = st.text_input("Classes (comma-separated)", "sec, nonsec", key="typed_classes")
                classes = sorted({c.strip() for c in typed.split(",") if c.strip()})
                if len(classes) < 2:
                    data_errors.append("Enter at least two classes.")

            with st.expander("Preview"):
                st.dataframe(train_df.head(50), hide_index=True, width="stretch")

            if classes:
                st.markdown("**What each class means** — shown to both models. Terse codes "
                            "like `PE` classify far better with a phrase.")
                editor = st.data_editor(
                    pd.DataFrame({"class": classes,
                                  "description": [KNOWN_DESCRIPTIONS.get(c, c) for c in classes]}),
                    hide_index=True, disabled=["class"], key=f"desc_{'|'.join(classes)}",
                    width="stretch",
                )
                descriptions = {r["class"]: (r["description"] or r["class"]).strip()
                                for _, r in editor.iterrows()}

    with right:
        st.subheader("Test data")
        holdout_on = st.checkbox("Hold out part of the training data", value=True, key="holdout_on")
        holdout = st.slider("Held-out share", 0.1, 0.4, 0.2, step=0.05, key="holdout",
                            disabled=not holdout_on)
        uploads = st.file_uploader("Upload one or more test files", accept_multiple_files=True,
                                   type=["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl"],
                                   key="test_uploads")
        for i, up in enumerate(uploads or []):
            with st.expander(f"📄 {up.name}", expanded=True):
                try:
                    raw = read_table(up)
                    text_col, label_col = column_picker(raw, f"test{i}", allow_no_label=False)
                    df = clean_dataset(raw, text_col, label_col)
                    tests.append(to_dataset(df, Path(up.name).stem))
                    note = f"{len(df)} rows"
                    if label_col and classes:
                        outside = sorted(set(df["label"]) - set(classes))
                        note += (f" · ⚠️ labels {outside} are not training classes, so this set "
                                 f"gets predictions but no scores") if outside else " · labels match"
                    st.caption(note)
                except Exception as exc:
                    st.error(f"Could not read {up.name}: {exc}")
        sample_tests = st.multiselect(
            "Sample test sets", [s for s in SAMPLE_DATASETS], key="sample_tests",
            default=["SecReq (CPN + GPS + ePurse)"] if classes == ["nonsec", "sec"] else [],
            help="The shipped samples are labelled sec / nonsec.",
        )
        for s in sample_tests:
            if source == "Sample dataset" and s == st.session_state.get("train_sample"):
                st.caption(f"Skipping {s}: it is the training set.")
                continue
            tests.append(to_dataset(load_sample(s), s.split(" (")[0]))

    # Assemble the training set and the held-out split.
    if train_df is not None and not data_errors:
        if holdout_on:
            stratify = train_df["label"] if "label" in train_df.columns else None
            tr, ho = train_test_split(train_df, test_size=holdout, stratify=stratify,
                                      random_state=int(seed))
            train_ds = to_dataset(tr.reset_index(drop=True), f"{train_name} (train)")
            tests.insert(0, to_dataset(ho.reset_index(drop=True), f"{train_name} (held-out)"))
        else:
            train_ds = to_dataset(train_df, train_name)
    elif train_df is None:
        data_errors.append("Choose a training dataset.")
    if not tests:
        data_errors.append("Add at least one test set (a held-out split, an upload or a sample).")


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

STAGE_SPAN = {  # overall progress bar share per pipeline stage
    "setup": (0.00, 0.03), "features_train": (0.03, 0.40), "features_test": (0.40, 0.60),
    "teacher": (0.60, 0.75), "train": (0.75, 0.98),
}

with tab_train:
    problems = list(data_errors) + list(config_errors)
    if not adapters:
        problems.append("Select at least one tree adapter.")
    if not supervisions:
        problems.append("Select at least one supervision signal.")
    if not demo and any(s.startswith("teacher") for s in supervisions) and needs_key:
        problems.append("Enter the Teacher's API key in the sidebar.")
    if train_ds is not None and not train_ds.has_labels and supervisions == ["ground_truth"]:
        problems.append("Unlabelled training data can only be trained on Teacher labels.")

    if train_ds is not None:
        n_eval = sum(len(t) for t in tests)
        teacher_used = any(s.startswith("teacher") for s in supervisions)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training rows", len(train_ds))
        c2.metric("Test rows", n_eval, help=", ".join(f"{t.name}: {len(t)}" for t in tests))
        c3.metric("Classes", len(classes or []))
        c4.metric("Models to train", len(adapters) * len(supervisions))
        if not demo:
            calls_per_text = 2 if student_cfg.debias and student_cfg.scoring == "logprobs" else 1
            student_calls = (len(train_ds) + n_eval) * calls_per_text
            teacher_calls = math.ceil(len(train_ds) / teacher_cfg.batch_size) if teacher_used else 0
            st.info(
                f"**Work estimate for a cold cache** — Student SLM: ~{student_calls:,} local "
                f"calls (~{student_calls * 0.13 / 60:.0f} min for qwen2.5:0.5b). "
                + (f"Teacher: ~{teacher_calls} calls, ~{len(train_ds) * 70:,} tokens"
                   + (f" (≈ ${len(train_ds) * 70 * 0.2 / 1e6:.3f} on OpenRouter)"
                      if "openrouter" in (teacher_cfg.base_url or "") else "") + ". "
                   if teacher_used else "No Teacher calls. ")
                + "Both are cached per text, so re-runs skip them."
            )

    if not demo and st.button("Test the Teacher connection", disabled=bool(config_errors) or not classes):
        with st.spinner(f"Asking {teacher_cfg.model}…"):
            try:
                probe_texts = (train_ds.texts[:3] if train_ds else
                               ["The system shall encrypt all stored passwords."])
                teacher = make_teacher(teacher_cfg, classes, descriptions)
                soft = teacher.generate_soft_labels(probe_texts)
                st.success(f"{teacher_cfg.model} answered: "
                           f"{teacher.usage['total_tokens']} tokens for {len(probe_texts)} texts.")
                st.dataframe(pd.DataFrame([{"text": t[:120], **{k: round(v, 3) for k, v in d.items()}}
                                           for t, d in zip(probe_texts, soft)]), hide_index=True)
            except Exception as exc:
                st.error(f"Teacher call failed: {exc}")

    for p in problems:
        st.warning(p)
    st.caption("Keep this tab open and leave the settings alone while a run is going — "
               "changing a widget restarts the page and stops the run (cached work is kept).")

    if st.button("🚀 Train and evaluate", type="primary", disabled=bool(problems)):
        status = st.status("Running the cascade…", expanded=True)
        bar = status.progress(0.0, text="Starting")
        log = status.empty()
        lines: list = []

        def progress(stage, fraction, message):
            lo, hi = STAGE_SPAN.get(stage, (0.98, 1.0))
            bar.progress(min(1.0, lo + (hi - lo) * fraction), text=message)
            if fraction in (0.0, 1.0) and (not lines or lines[-1] != message):
                lines.append(message)
                log.markdown("\n".join(f"- {l}" for l in lines[-8:]))

        try:
            run_dir = new_run_dir(f"{train_ds.name}_{'demo' if demo else student_cfg.model}")
            result = run_experiment(
                train_ds, tests, classes=classes, class_descriptions=descriptions,
                student_cfg=student_cfg, teacher_cfg=teacher_cfg, train_cfg=train_cfg,
                progress=progress, embedder=cached_embedder(train_cfg.embedder),
            )
            progress("save", 0.5, "Saving models, predictions and charts")
            save_run(result, run_dir)
            plots.save_all(result.payload(), run_dir / "plots")
            st.session_state["result"] = result
            st.session_state["demo_result"] = demo
            bar.progress(1.0, text="Done")
            status.update(label=f"Finished in {result.timings['total']:.0f}s", state="complete",
                          expanded=False)
            st.success(f"Saved to `{run_dir.relative_to(ROOT)}`. Open the **📊 Results** tab.")
        except Exception as exc:
            status.update(label="Run failed", state="error", expanded=True)
            st.error(f"{type(exc).__name__}: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

def payload_choices():
    choices = {}
    if "result" in st.session_state:
        res = st.session_state["result"]
        choices[f"This session · {res.run_dir.name if res.run_dir else 'unsaved'}"] = \
            ("session", res)
    for run in list_runs():
        if "result" in st.session_state and st.session_state["result"].run_dir == run:
            continue
        choices[f"Saved · {run.name}"] = ("disk", run)
    return choices


with tab_results:
    choices = payload_choices()
    if not choices:
        st.info("No runs yet. Train one in the 🚀 Train tab.")
    else:
        pick = st.selectbox("Run", list(choices), key="results_run")
        kind, obj = choices[pick]
        payload = obj.payload() if kind == "session" else json.loads((obj / "results.json").read_text())
        rows = pd.DataFrame(payload["results"])

        for w in payload["warnings"]:
            st.warning(w)

        diag = payload["diagnostics"]
        cfg = payload["config"]
        c1, c2, c3, c4 = st.columns(4)
        agree = diag.get("teacher_train_agreement")
        c1.metric("Teacher vs training labels", f"{agree['macro_f1']:.3f} macro F1" if agree else "—",
                  help="How often the Teacher agrees with your labels. Distilled adapters rarely beat it.")
        c2.metric("Teacher failures", diag.get("teacher_failed", "—"),
                  help="Texts the Teacher could not label; dropped before training.")
        c3.metric("Student confidence values", diag.get("student_distinct_confidences", "—"),
                  help="Distinct values in feature column 384. One value means a dead feature.")
        corr = diag.get("student_conf_correct_corr")
        c4.metric("Student confidence ↔ correctness", f"{corr:+.2f}" if corr is not None else "—")
        st.caption(f"Student `{cfg['student']['model']}` · Teacher `{cfg['teacher']['model']}` · "
                   f"embedder `{cfg['embedder']}` · {payload['train_size']} training rows · "
                   f"{payload['timings']['total']:.0f}s")

        if rows.empty:
            st.info("No test set had labels matching the training classes, so there are no "
                    "scores — predictions are still available below and in the 🔮 Predict tab.")
        else:
            st.subheader("Scores")
            table = rows.assign(
                adapter=rows["adapter"].map(DISPLAY_NAME),
                supervision=rows["supervision"].map(SUPERVISION_LABEL),
                lift_over_baseline=rows["macro_f1"] - rows["baseline_macro_f1"],
            )[["test_set", "adapter", "supervision", "macro_f1", "weighted_f1", "accuracy",
               "baseline_macro_f1", "student_macro_f1", "lift_over_baseline", "collapsed"]]
            st.dataframe(
                table.sort_values(["test_set", "macro_f1"], ascending=[True, False]),
                hide_index=True, width="stretch",
                column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                               ["macro_f1", "weighted_f1", "accuracy", "baseline_macro_f1",
                                "student_macro_f1", "lift_over_baseline"]},
            )
            best = rows.loc[rows.groupby("test_set")["macro_f1"].idxmax()]
            st.markdown("**Best per test set:** " + " · ".join(
                f"{r.test_set}: {DISPLAY_NAME[r.adapter]} on {SUPERVISION_LABEL[r.supervision]} "
                f"({r.macro_f1:.3f})" for r in best.itertuples()))

            p1, p2, p3, p4 = st.tabs(["Heatmap", "Bar charts", "Training curves", "Confusion matrices"])
            with p1:
                png = fig_png(plots.plot_macro_f1_heatmap(payload))
                st.image(png)
                st.download_button("Download heatmap PNG", png, "macro_f1_heatmap.png", "image/png")
            with p2:
                png = fig_png(plots.plot_macro_f1_bars(payload))
                st.image(png)
                st.download_button("Download bar chart PNG", png, "macro_f1_bars.png", "image/png")
            with p3:
                png = fig_png(plots.plot_training_curves(payload))
                st.image(png)
                st.caption("Solid = validation, dashed = training. The dot marks the tree count the "
                           "model was early-stopped at and is used for prediction.")
            with p4:
                test_pick = st.selectbox("Test set", list(dict.fromkeys(rows["test_set"])), key="cm_test")
                png = fig_png(plots.plot_confusion_matrices(payload, test_pick))
                st.image(png)

        st.subheader("Downloads")
        d1, d2 = st.columns(2)
        d1.download_button("results.json", json.dumps(payload, indent=2, default=str),
                           "results.json", "application/json")
        if kind == "session":
            preds = obj.predictions
        else:
            preds = {p.stem: pd.read_csv(p).to_dict("records")
                     for p in sorted((obj / "predictions").glob("*.csv"))}
        for name, recs in preds.items():
            d2.download_button(f"Predictions · {name} (CSV)", pd.DataFrame(recs).to_csv(index=False),
                               f"predictions_{name}.csv", "text/csv", key=f"dl_{pick}_{name}")


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────

with tab_predict:
    runs = list_runs()
    if not runs:
        st.info("Train a run first; saved runs appear here.")
    else:
        run_pick = st.selectbox("Trained run", runs, format_func=lambda p: p.name, key="pred_run")
        manifest = json.loads((run_pick / "models" / "manifest.json").read_text())
        run_is_demo = manifest["student"]["backend"] == "mock"
        results_path = run_pick / "results.json"
        ranking = {}
        if results_path.exists():
            res_rows = json.loads(results_path.read_text())["results"]
            for r in res_rows:
                ranking.setdefault(r["key"], []).append(r["macro_f1"])
        keys = sorted(manifest["models"], key=lambda k: -sum(ranking.get(k, [0])) / max(1, len(ranking.get(k, [1]))))
        model_pick = st.selectbox(
            "Model", keys, key="pred_model",
            format_func=lambda k: f"{DISPLAY_NAME[k.split('__')[0]]} · {SUPERVISION_LABEL[k.split('__')[1]]}"
            + (f" · mean test macro F1 {sum(ranking[k]) / len(ranking[k]):.3f}" if k in ranking else ""),
        )
        st.caption(f"Classes: {', '.join(manifest['classes'])} · needs Student "
                   f"`{manifest['student']['model']}` running locally, as at training time.")

        how = st.radio("Input", ["Type or paste", "Upload a file"], horizontal=True, key="pred_how")
        texts = []
        if how == "Type or paste":
            raw_text = st.text_area("One text per line", height=160, key="pred_text",
                                    placeholder="The system shall lock an account after five failed logins.")
            texts = [t.strip() for t in raw_text.splitlines() if t.strip()]
        else:
            up = st.file_uploader("File", type=["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl"],
                                  key="pred_upload")
            if up is not None:
                df = read_table(up)
                text_col = st.selectbox("Text column", list(df.columns),
                                        index=list(df.columns).index(guess_columns(df)[0]), key="pred_col")
                texts = clean_dataset(df, text_col)["text"].tolist()
                st.caption(f"{len(texts)} texts")

        if st.button("Classify", type="primary", disabled=not texts):
            try:
                cascade = cached_saved_cascade(str(run_pick), run_is_demo)
                bar = st.progress(0.0, text="Student SLM")
                labels, proba = cascade.predict(
                    texts, model_pick,
                    progress_callback=lambda d, t: bar.progress(d / t, text=f"Student SLM {d}/{t}"))
                bar.empty()
                out = pd.DataFrame({"text": texts, "prediction": labels,
                                    "confidence": proba.max(axis=1).round(4)})
                for i, c in enumerate(cascade.classes):
                    out[f"p({c})"] = proba[:, i].round(4)
                st.dataframe(out, hide_index=True, width="stretch")
                st.download_button("Download predictions (CSV)", out.to_csv(index=False),
                                   "predictions.csv", "text/csv")
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# About
# ─────────────────────────────────────────────────────────────────────────────

with tab_about:
    about_path = ROOT / "docs" / "how_it_works.md"
    if about_path.exists():
        st.markdown(about_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"`{about_path.relative_to(ROOT)}` is missing, so there is nothing to show here.")
