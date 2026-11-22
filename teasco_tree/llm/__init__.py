"""Frozen Student SLM (local Ollama) and the offline 70B Teacher."""

from teasco_tree.llm.base import BaseLLM
from teasco_tree.llm.mock import MockStudentLLM, MockTeacher
from teasco_tree.llm.ollama_student import OllamaLLM
from teasco_tree.llm.teacher import (
    TEACHER_PRESETS,
    DailyQuotaExhausted,
    OpenAITeacher,
    TeacherFailureError,
)

__all__ = [
    "BaseLLM", "MockStudentLLM", "MockTeacher", "OllamaLLM",
    "OpenAITeacher", "TEACHER_PRESETS", "TeacherFailureError", "DailyQuotaExhausted",
]
