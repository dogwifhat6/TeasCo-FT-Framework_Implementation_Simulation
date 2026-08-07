"""
Student module — lightweight classifiers that mimic the Teacher LLM.

- BaseStudent  : Logistic Regression trained on Teacher's soft labels.
- Corrector    : Random Forest / SVM trained on Base Student's hard examples.
"""

from teascoFT.student.base_student import BaseStudent
from teascoFT.student.corrector import Corrector

__all__ = ["BaseStudent", "Corrector"]
