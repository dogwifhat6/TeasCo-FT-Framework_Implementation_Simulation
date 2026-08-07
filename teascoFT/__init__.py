"""
TeasCo-FT: Knowledge Distillation + Cascade Architecture Framework
===================================================================

A production-ready framework for replacing expensive LLMs with fast,
specialized student models for text classification tasks.

Pipeline
--------
TRAINING:
  Raw Text → Feature Extractor → Teacher LLM (soft labels)
           → Base Student (trained via KD) → Corrector (trained on errors)

INFERENCE (LLM-free):
  Raw Text → Feature Extractor → Base Student
                                      │
                        (high conf) ──┤── (low conf) → Corrector
                                      │
                                Final Prediction

Quickstart
----------
    from teascoFT.distillation import KnowledgeDistillationTrainer
    from teascoFT.cascade import CascadePipeline
    from teascoFT.teacher import OpenAITeacher
    from teascoFT.features import TFIDFExtractor

    # 1. Set up teacher
    teacher = OpenAITeacher(api_key="sk-...", model="gpt-4o-mini")

    # 2. Choose a feature extractor
    extractor = TFIDFExtractor(max_features=20_000)

    # 3. Train the full student pipeline via knowledge distillation
    trainer = KnowledgeDistillationTrainer(teacher=teacher, extractor=extractor)
    trainer.fit(texts_train, y_train, classes=["positive", "negative", "neutral"])

    # 4. Save the pipeline (no LLM needed at inference)
    pipeline = trainer.build_pipeline()
    pipeline.save("my_classifier.pkl")

    # 5. Fast inference
    loaded = CascadePipeline.load("my_classifier.pkl")
    predictions = loaded.predict(["Some new text to classify"])
"""

from teascoFT.teacher import OpenAITeacher
from teascoFT.teacher.llm_teacher import BaseTeacher
from teascoFT.features import TFIDFExtractor, EmbeddingExtractor
from teascoFT.student import BaseStudent, Corrector
from teascoFT.distillation import KnowledgeDistillationTrainer
from teascoFT.cascade import CascadePipeline
from teascoFT.utils import macro_f1, classification_report_rich

__version__ = "0.1.0"
__all__ = [
    "OpenAITeacher",
    "BaseTeacher",
    "TFIDFExtractor",
    "EmbeddingExtractor",
    "BaseStudent",
    "Corrector",
    "KnowledgeDistillationTrainer",
    "CascadePipeline",
    "macro_f1",
    "classification_report_rich",
]
