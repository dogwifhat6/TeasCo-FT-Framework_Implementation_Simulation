## How it works

TeasCo-FT classifies text (for example, *security-related* vs *not*) with a teacher–student cascade in which the **only trainable part is a gradient-boosted tree ensemble**.

```
frozen Student SLM (local Ollama)  ─┐
                                    ├─> 385-dim features ─> tree adapter
sentence embedding (MiniLM, 384)   ─┘                        (XGBoost / CatBoost / LightGBM)
                                                              ^
70B Teacher LLM (offline, once) ── soft labels ──────────────┘
```

At inference time only the small local models run: MiniLM, the Student SLM and the trained trees.

### 1 · Features

Every text becomes a 385-dimensional vector:

| Columns | Source |
|---|---|
| 0–383 | `all-MiniLM-L6-v2` sentence embedding (the offline demo uses a hashing embedder instead) |
| 384 | the frozen Student SLM's confidence |

The Student's confidence is read from the **probability distribution of its first generated token**, not from a number the model states about itself. Asked to state a confidence, a 0.5B model answers about 1.0 for everything, which turns the feature into a constant. The options are shown as letters (`A) … B) …`) and asked **in both orders**, and the two answers are averaged. This cancels small models' strong preference for whichever option comes first.

Feature matrices are cached on disk under a hash of the embedder, the Student and the texts themselves, so changing any of them can never reuse stale features.

### 2 · Teacher

A 70B Teacher (Llama 3.3 70B Instruct by default) labels the **training texts once**, returning a probability distribution per text. Texts are sent in batches of 25 to save tokens, and results are cached per text, so a re-run costs nothing.

Teacher failures are never silent. A failed text would otherwise get a uniform distribution, and `argmax` turns a tie into a vote for the first class. So failed texts are dropped, and a run where more than 5% fail stops with an error. Rate limits are waited out using the provider's own retry hint, capped at 120 s. A spent *daily* quota stops the run immediately.

### 3 · Supervision signals

Each adapter can be trained on three signals:

| Signal | What the trees learn from |
|---|---|
| **Ground truth** | your dataset's own label column |
| **Teacher pseudo-labels** | the Teacher's top answer per text |
| **Teacher soft labels (KD)** | the Teacher's full distribution, i.e. knowledge distillation |

Trees only accept hard labels, so soft labels are fed in by **row expansion**. Each text becomes one copy per class, labelled with that class and weighted by the Teacher's probability for it. Weighted cross-entropy over the copies equals cross-entropy against the soft distribution, so the trees learn how sure the Teacher was, not just its answer.

Teacher-supervised models also validate against Teacher labels, so your gold labels never leak into a run that claims to learn from the Teacher alone.

### 4 · The gated update rule

The adapter should **learn only from its mistakes**. Boosted trees have no weights to backpropagate into, so the rule is applied in one of three ways (set per adapter under *Boosting settings*):

| Gate mode | How it works | Available for |
|---|---|---|
| `objective` | A custom loss gives confidently-correct rows zero gradient, so new trees can only fix errors. Re-checked every boosting round. | XGBoost |
| `staged` | Trees are added in blocks of 25. Before each block, rows the model already gets right are down-weighted to the *gate floor*. | all three |
| `none` | Plain boosting, as a control. | all three |

Two settings keep the gate from breaking training:

* **Gate margin (0.10).** A row is silenced only if it is correct *and* confidently so. With a strict gate, boosting's first round (where every row ties) silences a whole class and the model never learns anything.
* **Gate floor (0.10).** A floor of 0 retrains each block on only the current mistakes, which flips rows the model had right and makes validation F1 swing wildly between stages.

Every adapter uses early stopping on validation macro F1 and keeps the best tree count.

### 5 · Reading the results

Always compare a score against the two grey reference rows:

* **Majority-class baseline.** The macro F1 of predicting the most common class for everything. A model near this line has probably collapsed.
* **Student SLM zero-shot.** What the frozen Student manages on its own, with no training.

Three automatic checks flag silent failures as warnings:

1. the Student's confidence is identical for every text (a dead feature);
2. the Teacher gave every text the same label (Teacher supervision is skipped);
3. a trained model predicts a single class on a test set (marked **collapsed**).

### 6 · Saved runs

Every run is saved under `runs/<timestamp>_<name>/`:

* `results.json` with scores, diagnostics and settings (never the API key)
* `predictions/` with one CSV per test set
* `models/` with every trained model plus `manifest.json`
* `plots/` with every chart as a PNG

The **🔮 Predict** tab reloads a saved run to classify new text. It rebuilds the same features, so the same Student model must be available in Ollama.

### Offline demo

*Offline demo* mode replaces the Student and Teacher with keyword-based mock models and MiniLM with a hashing embedder. The whole flow runs in seconds with no Ollama server, API key or download, but **its scores mean nothing**.
