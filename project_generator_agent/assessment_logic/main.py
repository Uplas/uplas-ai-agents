# ... other imports ...
from .assessment_logic import format_submission_items_for_llm_prompt # New import

# ...

def construct_project_assessment_user_query(
    original_project: GeneratedProjectIdea,
    submission_items: List[ProjectSubmissionContentItem], # This model needs to be accessible
    user_profile: UserProfileSnapshotForProjects,
    language_code: str
) -> str:
    # Use the new helper
    submission_summary_str = format_submission_items_for_llm_prompt(submission_items)

    prompt_parts = [
        # ... (rest of the prompt construction as before) ...
        "\n--- User's Submission Details ---", # Changed from "Submission Summary" to match helper output
        submission_summary_str, # Use the formatted string
        # ... (rest of the prompt construction) ...
    ]
    return "\n".join(prompt_parts)

# ...
