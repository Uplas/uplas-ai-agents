# uplas-ai-agents/ttv_agent/main.py
# (Keeping existing imports and setup largely the same)
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import os
import uuid
import time
import httpx
import logging
import random # For attire selection and mock elements
import re # For simple SSML tag processing

# Existing imports from animation_logic
from .animation_logic.character_manager import ( #
    InstructorChars,
    get_character_avatar_id_from_service,
    get_character_attire_id,
    CharacterConfigError
)
from .animation_logic.avatar_api_client import ThirdPartyAvatarAPIClient, AvatarJobError #

# Assuming TTS Agent's request model if we need to specify SSML input
# from tts_agent.main import SynthesisInputType as TtsSynthesisInputType (if importing cross-agent models)
class TtsSynthesisInputType(str, Enum): # Local definition for clarity
    TEXT = "text"
    SSML = "ssml"

# --- Configuration (as per existing) ---
# ... (GCP_PROJECT_ID, TTV_GCS_BUCKET_NAME, DJANGO_TTV_CALLBACK_URL, etc.)
#
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
TTV_GCS_BUCKET_NAME = os.getenv("TTV_GCS_BUCKET_NAME")
DJANGO_TTV_CALLBACK_URL = os.getenv("DJANGO_TTV_CALLBACK_URL")
AI_TUTOR_AGENT_URL = os.getenv("AI_TUTOR_AGENT_URL", "http://localhost:8001")
TTS_AGENT_URL = os.getenv("TTS_AGENT_URL", "http://localhost:8002")
THIRD_PARTY_AVATAR_API_KEY = os.getenv("THIRD_PARTY_AVATAR_API_KEY")
THIRD_PARTY_AVATAR_BASE_URL = os.getenv("THIRD_PARTY_AVATAR_BASE_URL")
SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"] #
DEFAULT_LANGUAGE = "en-US" #


logging.basicConfig(level=logging.INFO) #
logger = logging.getLogger(__name__) #

# --- Pydantic Models (largely as per existing) ---
# UserProfileSnapshotForTTV, ContentSource, GenerateVideoRequest, etc.
#
class UserProfileSnapshotForTTV(BaseModel):
    industry: Optional[str] = Field(None, examples=["Healthcare"])
    profession: Optional[str] = Field(None, examples=["Nurse Practitioner"])
    # Add learning_pace if we want to use it for SSML break adjustments
    learning_pace_preference: Optional[str] = Field("normal", examples=["slow", "normal", "fast"])
    # ... other fields from existing model

class ContentSource(BaseModel):
    topic_id: Optional[str] = Field(None, examples=["topic_uuid_for_python_loops"])
    course_id: Optional[str] = Field(None, examples=["course_uuid_intro_python"])
    raw_text_content: Optional[str] = Field(None, examples=["Explain Python loops..."]) # This will likely be the NLP-processed script guide
    # Add a field for NLP-processed content URI if fetched separately
    # nlp_processed_content_gcs_uri: Optional[HttpUrl] = None
    # Or assume raw_text_content will contain the script with NLP tags.

    @validator('raw_text_content', always=True)
    def check_content_provided(cls, v, values): # As per existing
        if not v and not values.get('topic_id') and not values.get('course_id'):
            raise ValueError('Either topic_id/course_id or raw_text_content must be provided.')
        return v

class GenerateVideoRequest(BaseModel): # As per existing
    user_id: str
    content_source: ContentSource
    instructor_character: InstructorChars # Using the Enum
    user_profile_snapshot: UserProfileSnapshotForTTV
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)
    preferred_video_length_minutes: str = Field("3-5", description="Target length for the video script.")
    preferred_attire_name: Optional[str] = Field(None)
    additional_instructions: Optional[str] = Field(None)
    # NEW: Optional field for background selection strategy
    background_theme_preference: Optional[str] = Field(None, examples=["tech_office", "calm_library", "dynamic_abstract"])


class VideoGenerationJobStatus(str, Enum): # As per existing
    PENDING = "pending"
    # ... other statuses ...
    PREPARING_SCRIPT = "preparing_script" # New intermediate status
    GENERATING_SCRIPT = "generating_script"
    GENERATING_AUDIO = "generating_audio"
    SUBMITTING_TO_AVATAR_SERVICE = "submitting_to_avatar_service"
    RENDERING_AVATAR_VIDEO = "rendering_avatar_video"
    COMPLETED = "completed"
    FAILED = "failed"

# ... (GenerateVideoInitialResponse, VideoCallbackPayload as per existing)

# --- FastAPI Application Setup (as per existing) ---
app = FastAPI(title="Uplas Text-to-Video (TTV) Agent - Enhanced") #
video_jobs: Dict[str, Dict[str, Any]] = {} #

# Initialize Avatar Service Client (as per existing)
# This client's implementation in avatar_api_client.py is CRITICAL and placeholder.
#
if not THIRD_PARTY_AVATAR_API_KEY or not THIRD_PARTY_AVATAR_BASE_URL:
    logger.warning("THIRD_PARTY_AVATAR_API_KEY or THIRD_PARTY_AVATAR_BASE_URL not set. Avatar service calls will be mocked/fail.")
    avatar_service_client = ThirdPartyAvatarAPIClient(is_mock=True)
else:
    avatar_service_client = ThirdPartyAvatarAPIClient(
        api_key=THIRD_PARTY_AVATAR_API_KEY,
        base_url=THIRD_PARTY_AVATAR_BASE_URL
    )

# --- Helper: update_job_status_and_notify (as per existing) ---
# ...
async def update_job_status_and_notify(job_id: str, new_status: VideoGenerationJobStatus, **kwargs):
    # ... (implementation as in existing file)
    pass


# --- ENHANCED Background Task for Video Generation Pipeline ---
async def process_video_generation_task(job_id: str, request_data: GenerateVideoRequest): # Modified from
    await update_job_status_and_notify(job_id, VideoGenerationJobStatus.PREPARING_SCRIPT)
    
    raw_script_from_tutor: Optional[str] = None # This script should contain NLP tags
    processed_ssml_for_tts: Optional[str] = None
    # ... (other variables as in existing: generated_audio_gcs_url, tts_duration_seconds)

    try:
        # === Step 1: Generate Script (via AI Tutor Agent) ===
        # This step remains similar, but AI Tutor should now be aware of NLP-processed content
        # and its prompts should encourage it to utilize/retain the NLP tags.
        #
        logger.info(f"Job {job_id}: Requesting script from AI Tutor Agent.")
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.GENERATING_SCRIPT)
        # ... (existing AI Tutor call logic)
        # The tutor_payload should include context_ids that allow AI Tutor
        # to fetch NLP-processed content.
        # The `raw_script_from_tutor` is expected to contain the NLP tags.
        # For mock purposes:
        raw_script_from_tutor = (
            f"Hello {request_data.user_profile_snapshot.profession or 'learner'}. "
            "Today we discuss <topic>Topic X</topic>. "
            "This is <difficulty type=\"foundational_info\">very important</difficulty>. "
            "<analogy type=\"user_profile_analogy_placeholder\" /> "
            "Let's take a <pause strength=\"medium\" /> short break. "
            "Another key point is <emphasis level=\"strong\">this one</emphasis>! "
            "<visual_aid_suggestion type=\"diagram_needed\" description=\"Flowchart of process X\" />"
        )
        if not raw_script_from_tutor: raise ValueError("AI Tutor returned an empty script.")
        logger.info(f"Job {job_id}: Script generated. Length: {len(raw_script_from_tutor)}")
        video_jobs[job_id]["script_generated_preview"] = raw_script_from_tutor[:500]


        # === Step 1.5: SSML Preprocessing (NEW) ===
        # Convert our custom NLP tags (now in the script from AI Tutor) to actual SSML
        logger.info(f"Job {job_id}: Preprocessing script to SSML.")
        processed_ssml_for_tts = convert_script_to_ssml(
            raw_script_from_tutor,
            request_data.language_code,
            request_data.user_profile_snapshot.learning_pace_preference
        )
        # Extract visual aid suggestions for later use (conceptual)
        visual_aid_suggestions = extract_visual_aid_cues(raw_script_from_tutor)
        video_jobs[job_id]["visual_cues_identified"] = visual_aid_suggestions


        # === Step 2: Generate Audio (via TTS Agent - now with SSML) ===
        #
        logger.info(f"Job {job_id}: Requesting audio from TTS Agent using SSML.")
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.GENERATING_AUDIO)
        
        # Map TTV instructor to a TTS voice character (as in existing)
        tts_voice_char_map = { # Ensure these names align with your refined TTS map
            InstructorChars.SUSAN: "susan_us_standard", # Or _wavenet based on a quality flag
            InstructorChars.UNCLE_TREVOR: "trevor_us_standard"
        }
        # TODO: Add logic to select _wavenet if a 'high_quality_audio' flag is set in request_data
        tts_voice_character = tts_voice_char_map.get(request_data.instructor_character, "default_en_us_standard")

        tts_payload = {
            "input_type": TtsSynthesisInputType.SSML.value, # Specify SSML input
            "content_to_synthesize": processed_ssml_for_tts,
            "language_code": request_data.language_code,
            "voice_params": {"voice_character_name": tts_voice_character},
            "audio_config": {"audio_encoding": "MP3"}, # Or as per avatar service needs
            # "prefer_wavenet_quality": False # Add if you want to control this
        }
        # ... (existing TTS Agent call logic)
        # Mocked response:
        generated_audio_gcs_url = f"gs://{TTV_GCS_BUCKET_NAME}/mock_audio_{job_id}.mp3"
        tts_duration_seconds = 120.5
        if not generated_audio_gcs_url: raise ValueError("TTS Agent did not return an audio URL.")
        logger.info(f"Job {job_id}: Audio generated: {generated_audio_gcs_url}")


        # === Step 3: Generate Avatar Video ===
        #
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.SUBMITTING_TO_AVATAR_SERVICE)
        
        character_service_avatar_id = get_character_avatar_id_from_service(request_data.instructor_character.value)
        
        # ENHANCED Attire and Background Selection
        attire_tags = determine_attire_tags(request_data) # New helper
        chosen_attire_service_id = get_character_attire_id(
            instructor_character_name=request_data.instructor_character.value,
            preferred_attire_name=request_data.preferred_attire_name,
            tags=attire_tags
        )
        video_jobs[job_id]["attire_used"] = chosen_attire_service_id or "default_from_char_config"

        background_settings = determine_background_settings(request_data, visual_aid_suggestions) # New helper

        # Submit to the third-party avatar service
        # The `avatar_service_client.submit_video_creation_job` is where you'll
        # implement the specific API calls for your chosen avatar service.
        # It needs to be robust.
        logger.info(f"Job {job_id}: Submitting to Third-Party Avatar Service. Avatar: {character_service_avatar_id}, Attire: {chosen_attire_service_id}")
        # **** Call to avatar_service_client.submit_video_creation_job to be implemented by Mugambi HERE ****
        # service_job_details = await avatar_service_client.submit_video_creation_job(
        #     service_avatar_id=character_service_avatar_id,
        #     audio_file_gcs_url=generated_audio_gcs_url,
        #     service_attire_id=chosen_attire_service_id,
        #     language_code=request_data.language_code, # Some services might need this for lip-sync nuances
        #     background_settings=background_settings, # Pass dynamic background
        #     # custom_metadata={'uplas_job_id': job_id, 'user_id': request_data.user_id} # If service supports
        # )
        # Mocked response for now:
        service_job_details = {"service_job_id": f"avatar_svc_job_{uuid.uuid4()}", "initial_status": "queued"}
        # **** END Call ****
        
        third_party_job_id = service_job_details.get("service_job_id")
        # ... (rest of the polling logic as in existing, ensure robust error handling and backoff)
        #
        # The polling loop should be resilient.

        # === Step 4: Poll for Avatar Video Completion (existing logic) ===
        # ...

        # === Step 5: Finalize and Notify (existing logic) ===
        # ...

    except Exception as e: #
        logger.error(f"TTV Job {job_id}: Failed. Error: {e}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=str(e))

# NEW Helper function for SSML Preprocessing
def convert_script_to_ssml(raw_script: str, language_code: str, learning_pace: Optional[str]) -> str:
    logger.debug(f"Original script for SSML conversion: {raw_script[:200]}")
    # Simple replacements for InnovateAI custom tags to standard SSML
    # This should be made more robust with proper XML/HTML parsing if tags become complex
    ssml = raw_script
    
    # Emphasis
    ssml = re.sub(r'<emphasis level="strong">(.*?)</emphasis>', r'<emphasis level="strong">\1</emphasis>', ssml) # Google supports strong/moderate/reduced
    ssml = re.sub(r'<emphasis level="moderate">(.*?)</emphasis>', r'<emphasis level="moderate">\1</emphasis>', ssml)
    
    # Pauses (adjust ms based on learning_pace)
    pause_duration_ms = {"slow": "700ms", "normal": "400ms", "fast": "200ms"}
    selected_pause_ms = pause_duration_ms.get(learning_pace or "normal", "400ms")
    
    ssml = re.sub(r'<pause strength="medium" />', f'<break time="{selected_pause_ms}"/>', ssml)
    ssml = re.sub(r'<pause strength="long" />', f'<break time="{int(float(selected_pause_ms[:-2])*1.5)}ms"/>', ssml) # Example: long is 1.5x medium
    ssml = re.sub(r'<pause strength="short" />', f'<break time="{int(float(selected_pause_ms[:-2])*0.5)}ms"/>', ssml)

    # Remove our custom visual aid tags for TTS, they are for TTV logic
    ssml = re.sub(r'<visual_aid_suggestion.*?>.*?</visual_aid_suggestion>', '', ssml)
    ssml = re.sub(r'<visual_aid_suggestion.*?/>', '', ssml)
    # Remove difficulty tags for TTS
    ssml = re.sub(r'<difficulty.*?>', '', ssml)
    ssml = re.sub(r'</difficulty>', '', ssml)
    # Remove topic tags if any were left
    ssml = re.sub(r'<topic.*?>', '', ssml)
    ssml = re.sub(r'</topic>', '', ssml)
    # Remove analogy/example placeholder tags for TTS
    ssml = re.sub(r'<\/?(analogy|example).*?>', '', ssml)


    # Ensure the final output is wrapped in <speak> tags if not already (Google TTS prefers this for SSML)
    if not ssml.strip().startswith("<speak>"):
        ssml = f"<speak>{ssml}</speak>"
    logger.debug(f"Processed SSML: {ssml[:300]}")
    return ssml

# NEW Helper function to extract visual cues
def extract_visual_aid_cues(script_with_tags: str) -> List[Dict[str, str]]:
    cues = []
    # Example: <visual_aid_suggestion type="diagram_needed" description="Flowchart of process X" />
    pattern = r'<visual_aid_suggestion\s+type="([^"]+)"\s+description="([^"]+)"\s*/>'
    for match in re.finditer(pattern, script_with_tags):
        cues.append({"type": match.group(1), "description": match.group(2)})
    return cues

# NEW Helper function for attire selection strategy
def determine_attire_tags(request_data: GenerateVideoRequest) -> List[str]:
    tags = ["daily_professional"] # Default
    if request_data.additional_instructions:
        if "formal presentation" in request_data.additional_instructions.lower():
            tags = ["formal", "presentation"]
        elif "casual tutorial" in request_data.additional_instructions.lower():
            tags = ["casual", "tutorial", "friendly"]
    # Could also use request_data.content_source.topic_id to infer context for attire
    # e.g., if topic is about "Advanced Finance", add "formal_business" tag.
    logger.info(f"Determined attire tags: {tags}")
    return tags

# NEW Helper function for background selection strategy (conceptual)
def determine_background_settings(request_data: GenerateVideoRequest, visual_cues: List[Dict]) -> Optional[Dict]:
    # This depends heavily on what the third-party avatar service supports.
    # It might be a URL to an image/video, a predefined scene ID, or color codes.
    if request_data.background_theme_preference == "tech_office":
        return {"type": "image_url", "url": "gs://your-asset-bucket/backgrounds/tech_office.jpg"}
    if request_data.background_theme_preference == "calm_library":
        return {"type": "scene_id", "id": "library_scene_001"} # If service uses scene IDs
    
    # Potentially use visual_cues to pick a background if one matches a diagram description theme
    # e.g. if a visual_cue asks for a "stock chart", pick a financial background.
    
    logger.info("No specific background preference set, using avatar service default.")
    return None # Let avatar service use its default or character's default


# --- API Endpoints (largely as per existing) ---
# @app.post("/v1/generate-video", ...)
# @app.get("/v1/video-status/{job_id}", ...)
# @app.get("/health", ...)
# ... (These remain structurally similar to existing code)


# For asyncio.sleep in background task (as per existing)
import asyncio #

# if __name__ == "__main__": (as per existing)
# ...
