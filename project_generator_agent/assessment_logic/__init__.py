# uplas-ai-agents/project_generator_agent/assessment_logic/__init__.py
"""
InnovateAI: Assessment Logic Sub-package.
Contains utilities for parsing project submissions and processing assessment feedback
for the Personalized Real-World Project Generation & Assessment Agent.
"""
from .submission_parser import format_submission_items_for_llm_prompt

__all__ = [
    "format_submission_items_for_llm_prompt"
]
