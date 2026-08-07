"""
train_example.py
================
End-to-end training example using the 20 Newsgroups dataset.

This script demonstrates the complete TeasCo-FT pipeline:
  1. Load a real-world dataset (no external API needed for the demo — we mock
     Teacher labels using a RandomForest as stand-in to avoid API costs).
  2. Extract TF-IDF features.
  3. Train the Base Student via Knowledge Distillation.
  4. Train the Corrector on hard examples.
  5. Evaluate the full Cascade Pipeline.
  6. Save the model.

To use a real Teacher LLM, replace MockTeacher with OpenAITeacher.

Usage
-----
    python examples/train_example.py

    # With real OpenAI teacher:
    OPENAI_API_KEY=sk-... python examples/train_example.py --use-llm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── Make teascoFT importable from examples/ ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from teascoFT import (
    KnowledgeDistillationTrainer,
    TFIDFExtractor,
    OpenAITeacher,
    macro_f1,
    classification_report_rich,
)
from teascoFT.teacher.llm_teacher import BaseTeacher


# ─────────────────────────────────────────────────────────────────────────────
# Mock Teacher (for demo / testing without an API key)
# ─────────────────────────────────────────────────────────────────────────────

class MockTeacher(BaseTeacher):
    """
    A mock Teacher that uses a RandomForest as a stand-in for the LLM.
    Useful for testing the pipeline without an API key.

    In production, replace this with OpenAITeacher.
    """

    def __init__(self, class_list: List[str]) -> None:
        self._classes = class_list
        self._rf = None

    @property
    def classes(self) -> List[str]:
        return self._classes

    def fit_on_data(self, X, y: List[str]) -> None:
        """Internal method to pre-train the mock teacher on raw features."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        self._le = LabelEncoder().fit(self._classes)
        self._rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        self._rf.fit(X, self._le.transform(y))

    def generate_soft_labels(
        self,
        texts: List[str],
        classes: List[str] = None,
    ) -> List[Dict[str, float]]:
        """Return soft labels from RandomForest probability estimates."""
        if self._rf is None:
            raise RuntimeError("MockTeacher must be pre-fitted. Call fit_on_data() first.")
        # We can't transform texts here — so we use a workaround:
        # This is called after the extractor fits, so we use stored vectors.
        # In practice, the real teacher calls the LLM API on raw text.
        proba = self._rf.predict_proba(self._X_cache)
        target = classes if classes else self._classes
        return [
            {cls: float(proba[i, j]) for j, cls in enumerate(self._classes)}
            for i in range(len(texts))
        ]

    def set_X_cache(self, X) -> None:
        """Store feature vectors so generate_soft_labels can use them."""
        self._X_cache = X


# ─────────────────────────────────────────────────────────────────────────────
# Main Script
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TeasCo-FT Training Example")
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Use real OpenAI Teacher (requires OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model to use as Teacher (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--output", default="models/newsgroups_classifier.pkl",
        help="Path to save the trained pipeline"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Confidence threshold for Base Student → Corrector routing (default: 0.65)"
    )
    parser.add_argument(
        "--categories", nargs="+",
        default=["sci.med", "sci.space", "rec.sport.hockey", "talk.politics.guns"],
        help="20 Newsgroups categories to use"
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  TeasCo-FT: Knowledge Distillation + Cascade Pipeline")
    print("=" * 65 + "\n")

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"📂 Loading 20 Newsgroups ({', '.join(args.categories)})...")
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split

    data_train = fetch_20newsgroups(
        subset="train", categories=args.categories,
        remove=("headers", "footers", "quotes"),
    )
    data_test = fetch_20newsgroups(
        subset="test", categories=args.categories,
        remove=("headers", "footers", "quotes"),
    )

    # Map integer targets back to category names
    target_names = data_train.target_names
    texts_train = data_train.data
    y_train = [target_names[t] for t in data_train.target]
    texts_test = data_test.data
    y_test = [target_names[t] for t in data_test.target]

    print(f"  Train: {len(texts_train)} samples | Test: {len(texts_test)} samples")
    print(f"  Classes: {target_names}\n")

    # ── Feature Extractor ─────────────────────────────────────────────────
    extractor = TFIDFExtractor(max_features=30_000, ngram_range=(1, 2))

    # ── Teacher ───────────────────────────────────────────────────────────
    if args.use_llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not set.")
            sys.exit(1)
        teacher = OpenAITeacher(
            api_key=api_key,
            model=args.model,
            classes=target_names,
        )
        soft_labels = None  # Will be generated by the trainer
        print(f"🤖 Teacher: OpenAI {args.model}\n")
    else:
        print("🤖 Teacher: MockTeacher (RandomForest stand-in — no API needed)\n")

        # For MockTeacher, we pre-fit on training features
        mock_teacher = MockTeacher(class_list=target_names)
        X_for_mock = TFIDFExtractor(max_features=30_000).fit_transform(texts_train)
        mock_teacher.fit_on_data(X_for_mock, y_train)
        mock_teacher.set_X_cache(X_for_mock)

        # Pre-generate soft labels so the trainer doesn't call generate_soft_labels
        # (which would need the trained extractor's transform)
        print("  Generating mock soft labels...")
        soft_labels_train = mock_teacher.generate_soft_labels(texts_train)
        teacher = mock_teacher

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = KnowledgeDistillationTrainer(
        teacher=teacher,
        extractor=extractor,
        confidence_threshold=args.threshold,
        corrector_model_type="svm",  # SVM works well with TF-IDF sparse features
        base_student_C=5.0,
        verbose=True,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("🚀 Starting Knowledge Distillation Training...\n")

    if args.use_llm:
        pipeline = trainer.fit(texts_train, y_train)
    else:
        # Pass pre-computed soft labels to avoid redundant mock calls
        pipeline = trainer.fit(
            texts_train, y_train,
            classes=target_names,
            soft_labels=soft_labels_train,
        )

    # ── Evaluate on Test Set ──────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("📊 Evaluation on Hold-Out Test Set")
    print("─" * 55)
    results = pipeline.evaluate(texts_test, y_test, verbose=True)

    print(f"\n  Test Macro F1  : {results['macro_f1']:.4f}")
    print(f"  Test Accuracy  : {results['accuracy']:.4f}")

    # ── Save Pipeline ─────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(output_path)
    print(f"\n✅ Done! Pipeline saved to: {output_path.resolve()}")
    print("   Use inference_example.py to load and run predictions.\n")


if __name__ == "__main__":
    main()
