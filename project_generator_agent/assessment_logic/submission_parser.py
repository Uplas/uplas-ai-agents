# uplas-ai-agents/project_generator_agent/assessment_logic/submission_parser.py
from typing import List, TYPE_CHECKING
import logging

# InnovateAI Note on Type Hinting & Circular Dependencies:
# To avoid circular imports if ProjectSubmissionContentItem is defined in main.py,
# we use TYPE_CHECKING to allow type hints without runtime import issues.
# A better long-term solution is to move Pydantic models to a dedicated `models.py`
# within the agent's directory or to `shared_ai_libs` if used by multiple agents.
if TYPE_CHECKING:
    from ..main import ProjectSubmissionContentItem, ProjectSubmissionContentType # Relative import assuming main.py is one level up

logger = logging.getLogger(__name__)

MAX_CONTENT_PREVIEW_LENGTH = 400 # Characters for inline preview in prompt

def format_submission_items_for_llm_prompt(
    submission_items: List['ProjectSubmissionContentItem']
) -> str:
    """
    InnovateAI: Formats a list of project submission items into a string
    suitable for inclusion in an LLM prompt for assessment.

    Provides a preview for textual content and indicates the type and location
    for URL-based submissions.

    Args:
        submission_items: A list of ProjectSubmissionContentItem objects.

    Returns:
        A formatted string summarizing the submission items.
    """
    if not submission_items:
        return "No submission items provided by the user."

    formatted_parts = ["\n--- User's Submission Details ---"]
    for i, item in enumerate(submission_items):
        item_summary = f"\nSubmission Item {i+1}:"
        item_summary += f"\n- Type: {item.content_type.value}"
        if item.filename:
            item_summary += f"\n- Filename: {item.filename}"
        if item.notes:
            item_summary += f"\n- User Notes: {item.notes}"

        # Provide content preview or URL
        # InnovateAI Note: For GCS URLs or GitHub URLs, the LLM won't be able to access them directly.
        # The prompt should instruct the LLM to assess based on the *description* of these items
        # or any textual summary provided. If actual content analysis of URL-based items is needed,
        # a pre-processing step would have to fetch and summarize that content into text first.
        # For this version, we focus on how the LLM is informed about what was submitted.

        # Using item.content_type requires ProjectSubmissionContentType to be resolved
        # For now, we access value directly from the Pydantic model if it's available
        content_type_value = item.content_type.value if hasattr(item.content_type, 'value') else str(item.content_type)

        if "url" in content_type_value.lower() or "github" in content_type_value.lower():
            item_summary += f"\n- Location: {item.value}"
            item_summary += "\n- Content Note: Assessor should evaluate based on the nature of the linked content (e.g., code in repo, design in ZIP)."
        elif content_type_value in ["text_report", "python_code_string", "markdown_document"]:
            preview = item.value[:MAX_CONTENT_PREVIEW_LENGTH]
            if len(item.value) > MAX_CONTENT_PREVIEW_LENGTH:
                preview += "..."
            item_summary += f"\n- Content Preview (first {MAX_CONTENT_PREVIEW_LENGTH} chars):\n{preview}"
        else:
            item_summary += f"\n- Content: Data provided (type: {item.content_type.value})"

        formatted_parts.append(item_summary)

    if not formatted_parts: # Should not happen if submission_items is not empty
        return "User submission details were provided but could not be formatted."

    return "\n".join(formatted_parts)

# InnovateAI Future Enhancement for this module:
# async def fetch_and_summarize_gcs_content(gcs_url: str) -> str:
#     """
#     If the LLM needs actual content from GCS URLs, this function would
#     download the content and summarize it (e.g., text extraction from PDF,
#     tree listing for ZIP, head of code files). This is a more complex task.
#     For now, the LLM assesses based on the *fact* that a GCS URL was submitted
#     and any accompanying textual descriptions.
#     """
#     logger.info(f"InnovateAI Placeholder: Would fetch and summarize GCS content from {gcs_url}")
#     return f"Content from {gcs_url} was (conceptually) fetched. It appears to be a design document."
