"""
Teacher module — wraps LLM providers and exposes a unified interface
for generating soft label probability distributions over text classes.
"""

from teascoFT.teacher.llm_teacher import BaseTeacher, OpenAITeacher

__all__ = ["BaseTeacher", "OpenAITeacher"]
