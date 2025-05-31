# uplas-ai-agents/nlp_content_agent/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import os
import uuid
import time
import httpx # For any potential future internal calls, though not primary for this agent
import logging
import json

# Assuming shared_ai_libs is accessible in the Python path
# from shared_ai_libs.main import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
# For now, defining them here if shared_ai_libs isn't set up in the execution path for this new agent
SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"]
DEFAULT_LANGUAGE = "en-US"


# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
NLP_LLM_MODEL_NAME = os.getenv("NLP_LLM_MODEL_NAME", "gemini-1.5-flash-001") # Or your preferred Gemini model for NLP tasks

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for NLP Agent Output (Our Structured Educational Gold) ---

class NlpTopic(BaseModel):
    topic_id: str = Field(default_factory=lambda: f"topic_{uuid.uuid4().hex[:8]}")
    topic_title: str = Field(..., examples=["Understanding Superposition"])
    key_concepts: List[str] = Field(default_factory=list, examples=[
        "Qubits can represent 0, 1, or a combination of both.",
        "Superposition allows quantum computers to perform many calculations at once."
    ])
    # This content will have our XML-like tags for analogies, examples, etc.
    content_with_tags: str = Field(..., examples=["A classical bit is either 0 or 1. <analogy type=\"comparison_to_classical_needed\" /> ..."])
    # Optional: Add estimated reading time or complexity if LLM can provide
    estimated_complexity_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class NlpLesson(BaseModel):
    lesson_id: str = Field(default_factory=lambda: f"lesson_{uuid.uuid4().hex[:8]}")
    lesson_title: str = Field(..., examples=["What is a Qubit?"])
    # Optional: A brief summary of the lesson if LLM can generate it
    lesson_summary: Optional[str] = None
    topics: List[NlpTopic] = Field(default_factory=list)


class ProcessedModule(BaseModel):
    module_id: str = Field(..., examples=["course101_module3"])
    module_title: Optional[str] = Field(None, examples=["Introduction to Quantum Computing"])
    language_code: str = Field(..., examples=["en-US"])
    lessons: List[NlpLesson] = Field(default_factory=list)
    processing_time_ms: Optional[float] = None
    llm_model_used: Optional[str] = None

# --- Pydantic Models for API Request ---

class ProcessContentRequest(BaseModel):
    module_id: str = Field(..., examples=["course101_module3_raw"])
    raw_text_content: str = Field(..., min_length=100, description="The full raw text content of the course module.")
    language_code: str = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)
    module_title: Optional[str] = Field(None, description="Optional title for the module if known.")

    @validator('language_code')
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language_code '{v}' in ProcessContentRequest. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

# --- Vertex AI LLM Client Logic (Inspired by other agents) ---
class VertexAILLMClientForNLP:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Actual Vertex AI client initialization happens within methods or globally if preferred
        # For this example, assuming it's handled similarly to your other agents.
        # from google.cloud import aiplatform # Ensure this is imported
        # if GCP_PROJECT_ID:
        #     aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        # else:
        #     logger.warning("GCP_PROJECT_ID not set, Vertex AI LLM client might not function.")


    async def _call_gemini_api(self, system_prompt: str, user_query: str, is_json_output: bool = True) -> str:
        """
        Placeholder for actual call to Gemini API.
        You will replace this with your robust Gemini API call logic.
        """
        logger.info(f"Simulating Gemini API call for model: {self.model_name}")
        logger.info(f"System Prompt (sample): {system_prompt[:200]}...")
        logger.info(f"User Query (sample): {user_query[:200]}...")

        # **** START Call to actual Gemini API to be implemented by Mugambi HERE ****
        # Example structure of what your real call might involve:
        # from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
        # model = GenerativeModel(self.model_name, system_instruction=[Part.from_text(system_prompt)])
        # generation_config = GenerationConfig(
        #     temperature=0.2, # Lower temp for more deterministic structuring
        #     max_output_tokens=4096, # Adjust as needed
        # )
        # if is_json_output:
        #     generation_config.response_mime_type = "application/json"
        #     # If Gemini model supports schema for response_schema, pass it here.
        #     # For simplicity, assuming the prompt clearly asks for JSON.
        #
        # response = await model.generate_content_async(
        #     [Part.from_text(user_query)],
        #     generation_config=generation_config
        # )
        # response_text = "".join([part.text for part in response.candidates[0].content.parts if part.text])
        # return response_text
        # **** END Call to actual Gemini API to be implemented by Mugambi HERE ****

        # Mocked responses for now based on the task:
        if "segment it into distinct, high-level lessons" in system_prompt:
            # Mock for macro-segmentation
            mock_lessons_data = [
                {"lesson_title": "Lesson 1: The Basics", "text_segment_start_index": 0, "text_segment_end_index": 150},
                {"lesson_title": "Lesson 2: Advanced Topics", "text_segment_start_index": 151, "text_segment_end_index": 300}
            ]
            return json.dumps(mock_lessons_data)
        elif "Your tasks are:" in system_prompt and "Identify and list the core topics" in system_prompt:
            # Mock for micro-segmentation and enrichment
            mock_lesson_detail = {
              "lesson_title": "Mocked Lesson Title (from input)", # This would be dynamic
              "topics": [
                {
                  "topic_title": f"Mock Topic 1 in {user_query.split('language: ')[1].split(']')[0] if 'language: ' in user_query else DEFAULT_LANGUAGE}",
                  "key_concepts": ["Mock Concept A", "Mock Concept B"],
                  "content_with_tags": "This is mock content for Topic 1. <analogy type=\"everyday_analogy_needed\" /> It needs an example. <example domain=\"general_example_needed\" /> Is this clear? <interactive_question_opportunity text_suggestion=\"What was the main point of Topic 1?\" /> <visual_aid_suggestion type=\"simple_diagram\" description=\"Diagram for Topic 1\" /> <difficulty type=\"foundational_info\" />"
                }
              ]
            }
            return json.dumps(mock_lesson_detail)
        return "{}" # Default empty JSON

    async def macro_segment_module(self, full_text: str, language_code: str, module_title: Optional[str]) -> List[Dict[str, Any]]:
        """
        Uses LLM to break the module into high-level lessons.
        Returns a list of lesson titles and their corresponding text segments (or indices).
        """
        system_prompt = (
            f"You are an expert instructional designer. Your task is to segment the provided course module text, which is in [{language_code}], "
            f"into distinct, high-level lessons. For each lesson, provide a concise, engaging title in [{language_code}] and the character start and end indices from the original text that constitute that lesson. "
            "The module title is '{module_title if module_title else 'N/A'}'. Consider logical breaks in content, flow, and typical lesson lengths. "
            "Respond ONLY with a valid JSON list of objects. Each object must have 'lesson_title' (string), 'text_segment_start_index' (integer), and 'text_segment_end_index' (integer)."
        )
        user_query = f"Here is the module text in [{language_code}] to segment:\n\n{full_text}"

        try:
            response_str = await self._call_gemini_api(system_prompt, user_query, is_json_output=True)
            lessons_data = json.loads(response_str)
            if not isinstance(lessons_data, list): # Basic validation
                raise ValueError("LLM response for macro-segmentation was not a list.")
            # Further validation for keys like 'lesson_title', 'text_segment_start_index', 'text_segment_end_index' should be done here.
            return lessons_data
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError in macro_segment_module: {e}. Response: {response_str[:500]}")
            raise ValueError("Failed to parse LLM response for macro-segmentation.")
        except Exception as e:
            logger.error(f"Error in macro_segment_module: {e}", exc_info=True)
            raise

    async def micro_segment_and_enrich_lesson(self, lesson_text: str, lesson_title_from_macro: str, language_code: str) -> Dict[str, Any]:
        """
        Uses LLM to break a lesson into topics, identify key concepts, and tag for enrichments.
        """
        system_prompt = (
            f"You are an AI pedagogical expert. Your current task is to analyze the following lesson text snippet which is part of a lesson titled '{lesson_title_from_macro}'. The text is in [{language_code}]. "
            "Your objectives are to:"
            "1. Divide this lesson snippet into logical topics. For each topic, provide a concise 'topic_title' in [{language_code}]."
            "2. For each topic, extract 2-4 crucial 'key_concepts' as a list of strings, also in [{language_code}]."
            "3. For each topic, provide the 'content_with_tags'. This is the original text for the topic, but you MUST intelligently intersperse it with XML-like tags where appropriate: "
            "   - `<analogy type=\"[general_analogy_needed|technical_analogy_needed|user_profile_analogy_placeholder]\" />` for complex ideas."
            "   - `<example domain=\"[general_example_needed|finance_example_needed|tech_example_needed|user_profile_example_placeholder]\" />` for illustrating points."
            "   - `<interactive_question_opportunity text_suggestion=\"[Suggest a brief checking question in [language_code] here]\" />` at points good for engagement."
            "   - `<visual_aid_suggestion type=\"[diagram_needed|chart_needed|animation_cue]\" description=\"[Briefly describe visual in [language_code]]\" />` for concepts best shown visually."
            "   - `<difficulty type=\"[foundational_info|intermediate_detail|advanced_detail]\" />` to classify content sections."
            "Ensure the tags are placed naturally within the flow of the text. All text content within your JSON output (titles, concepts, tag suggestions) must be in [{language_code}]."
            "Respond ONLY with a single, valid JSON object adhering to this structure: "
            "{'lesson_title': '[The provided lesson_title_from_macro]', 'topics': [{'topic_title': '...', 'key_concepts': ['...', '...'], 'content_with_tags': 'Text with <tags/> ...'}, ...]}"
        )
        # Pass the language code explicitly in the user query as well for the mock response logic, or ensure your real Gemini call uses it.
        user_query = f"Analyze this lesson text in [{language_code}]:\n\n{lesson_text}"


        try:
            response_str = await self._call_gemini_api(system_prompt, user_query, is_json_output=True)
            enriched_lesson_data = json.loads(response_str)
            # Basic validation
            if not isinstance(enriched_lesson_data, dict) or "topics" not in enriched_lesson_data:
                raise ValueError("LLM response for micro-segmentation was not a valid lesson structure.")
            return enriched_lesson_data
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError in micro_segment_and_enrich_lesson: {e}. Response: {response_str[:500]}")
            raise ValueError("Failed to parse LLM response for micro-segmentation.")
        except Exception as e:
            logger.error(f"Error in micro_segment_and_enrich_lesson: {e}", exc_info=True)
            raise

# Initialize client
nlp_llm_client = VertexAILLMClientForNLP(model_name=NLP_LLM_MODEL_NAME)

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas NLP Content Structuring & Augmentation Agent",
    description="Processes raw course content into structured, enriched learning units using Vertex AI.",
    version="0.1.0" # Initial version
)

@app.post("/v1/process-course-content", response_model=ProcessedModule, status_code=status.HTTP_200_OK)
async def process_content_endpoint(request_data: ProcessContentRequest, background_tasks: BackgroundTasks):
    """
    Accepts raw course content, processes it through NLP pipeline,
    and returns a structured, enriched version.
    """
    start_time = time.perf_counter()
    logger.info(f"Received request to process module_id: {request_data.module_id} in language: {request_data.language_code}")

    if not GCP_PROJECT_ID: # Ensure critical configs are present
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="NLP service is not properly configured (missing GCP Project ID).")

    final_lessons: List[NlpLesson] = []

    try:
        # Stage 1: Macro-segmentation
        logger.info(f"Starting macro-segmentation for module: {request_data.module_id}")
        macro_segments = await nlp_llm_client.macro_segment_module(
            request_data.raw_text_content,
            request_data.language_code,
            request_data.module_title
        )
        logger.info(f"Macro-segmentation complete. Found {len(macro_segments)} potential lessons.")

        # Stage 2: Micro-segmentation & Enrichment for each lesson segment
        for i, segment_info in enumerate(macro_segments):
            lesson_title_from_macro = segment_info.get("lesson_title", f"Lesson {i+1}")
            start_idx = segment_info.get("text_segment_start_index")
            end_idx = segment_info.get("text_segment_end_index")

            if start_idx is None or end_idx is None or start_idx >= end_idx :
                logger.warning(f"Skipping invalid segment: {segment_info} for module {request_data.module_id}")
                continue

            lesson_text_snippet = request_data.raw_text_content[start_idx:end_idx]
            if not lesson_text_snippet.strip():
                logger.info(f"Skipping empty lesson text snippet for lesson: {lesson_title_from_macro}")
                continue

            logger.info(f"Starting micro-segmentation for lesson: '{lesson_title_from_macro}'")
            enriched_lesson_data_dict = await nlp_llm_client.micro_segment_and_enrich_lesson(
                lesson_text_snippet,
                lesson_title_from_macro,
                request_data.language_code
            )
            
            # Map LLM output to our Pydantic NlpLesson model
            # The enriched_lesson_data_dict is expected to match the structure requested in the prompt
            # (lesson_title, topics list with topic_title, key_concepts, content_with_tags)
            
            current_lesson_topics = []
            for topic_data in enriched_lesson_data_dict.get("topics", []):
                current_lesson_topics.append(NlpTopic(
                    topic_title=topic_data.get("topic_title", "Untitled Topic"),
                    key_concepts=topic_data.get("key_concepts", []),
                    content_with_tags=topic_data.get("content_with_tags", "")
                    # estimate_complexity_score can be added if LLM provides it
                ))
            
            # Ensure the lesson title from micro-segmentation (if returned and different) or macro is used.
            # The prompt for micro_segment_and_enrich_lesson asks LLM to include 'lesson_title_from_macro'.
            final_lesson_title = enriched_lesson_data_dict.get("lesson_title", lesson_title_from_macro)

            final_lessons.append(NlpLesson(
                lesson_title=final_lesson_title,
                topics=current_lesson_topics
                # lesson_summary can be added if LLM provides it
            ))
            logger.info(f"Micro-segmentation for lesson '{final_lesson_title}' complete.")

    except ValueError as ve: # Catch parsing errors or validation errors from our logic
        logger.error(f"ValueError during content processing for {request_data.module_id}: {ve}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing content: {str(ve)}")
    except Exception as e:
        logger.error(f"Unexpected error during content processing for {request_data.module_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during content processing.")

    if not final_lessons:
        logger.warning(f"No lessons were successfully processed for module: {request_data.module_id}")
        # Decide if this is an error or just an empty result
        # For now, let's return an empty list of lessons if nothing was processed.

    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000

    logger.info(f"Successfully processed module_id: {request_data.module_id}. Time taken: {processing_time_ms:.2f} ms")
    return ProcessedModule(
        module_id=request_data.module_id, # Or a new ID for the processed version
        module_title=request_data.module_title,
        language_code=request_data.language_code,
        lessons=final_lessons,
        processing_time_ms=round(processing_time_ms, 2),
        llm_model_used=nlp_llm_client.model_name
    )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if not GCP_PROJECT_ID or not NLP_LLM_MODEL_NAME:
        return {"status": "unhealthy", "reason": "Required configurations (GCP_PROJECT_ID, NLP_LLM_MODEL_NAME) are missing.", "service": "NLP_Content_Agent"}
    return {"status": "healthy", "service": "NLP_Content_Agent"}

if __name__ == "__main__":
    import uvicorn
    # Ensure environment variables are set for local testing
    # Example: GCP_PROJECT_ID, NLP_LLM_MODEL_NAME
    if not GCP_PROJECT_ID:
        print("Warning: GCP_PROJECT_ID is not set. Please set this environment variable for the NLP agent.")
    
    # Default port, e.g., 8005 or as configured via PORT env var
    port = int(os.getenv("PORT", 8005))
    uvicorn.run(app, host="0.0.0.0", port=port)
