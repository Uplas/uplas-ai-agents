# uplas-ai-agents/ttv_agent/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import os
import uuid
import time
import httpx # For making API calls to other agents and Django
import logging
import random # For attire selection and mock elements
import re # For SSML and tag processing
import asyncio # For asyncio.sleep

# Assuming character_manager is in animation_logic, and avatar_api_client is also there
# These imports are based on the structure seen in the uploaded files.
from .animation_logic.character_manager import (
    InstructorChars, # Enum for character names
    get_character_config,
    get_avatar_service_id as get_character_avatar_id_from_service, # Service-specific ID for the character model
    get_voice_settings as get_character_voice_settings, # Default voice settings for character (can be overridden)
    get_attire_id as get_character_attire_id, # Service-specific ID for attire
    CharacterConfigError
)
# This will be our refined ThirdPartyAvatarAPIClient interface
from .animation_logic.avatar_api_client import ThirdPartyAvatarAPIClient, AvatarJobError


# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
TTV_GCS_BUCKET_NAME = os.getenv("TTV_GCS_BUCKET_NAME") # For storing final videos/thumbnails
DJANGO_TTV_CALLBACK_URL = os.getenv("DJANGO_TTV_CALLBACK_URL") # e.g., http://your-django-app/api/internal/ttv-callback/

# URLs for other Uplas AI Agents
AI_TUTOR_AGENT_URL = os.getenv("AI_TUTOR_AGENT_URL", "http://localhost:8001") # Default for local dev
TTS_AGENT_URL = os.getenv("TTS_AGENT_URL", "http://localhost:8002") # Default for local dev

# API Key for the Third-Party Avatar Service
THIRD_PARTY_AVATAR_API_KEY = os.getenv("THIRD_PARTY_AVATAR_API_KEY")
THIRD_PARTY_AVATAR_BASE_URL = os.getenv("THIRD_PARTY_AVATAR_BASE_URL") # If needed by the client

# Supported languages - align with other agents
SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"]
DEFAULT_LANGUAGE = "en-US"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API Contract (Updated) ---

# InnovateAI Enhancement: Added learning_pace_preference for dynamic SSML generation
class UserProfileSnapshotForTTV(BaseModel): #
    industry: Optional[str] = Field(None, examples=["Healthcare"])
    profession: Optional[str] = Field(None, examples=["Nurse Practitioner"])
    country: Optional[str] = Field(None, examples=["Canada"])
    city: Optional[str] = Field(None, examples=["Toronto"])
    career_interest: Optional[str] = Field(None, examples=["AI in Medicine"])
    learning_goals: Optional[str] = Field(None, examples=["Understand how AI can improve diagnostics."])
    preferred_tutor_persona: Optional[str] = Field("Supportive and clear", examples=["Socratic", "Technical"])
    learning_pace_preference: Optional[str] = Field("normal", examples=["slow", "normal", "fast"], description="User's preferred learning pace to adjust video timing.")


class ContentSource(BaseModel): #
    topic_id: Optional[str] = Field(None, examples=["topic_uuid_for_python_loops"])
    course_id: Optional[str] = Field(None, examples=["course_uuid_intro_python"])
    # This raw_text_content should ideally be the script from AI Tutor, potentially with NLP enrichment tags.
    raw_text_content: Optional[str] = Field(None, examples=["Explain Python loops with an example for a beginner in finance."])

    @validator('raw_text_content', always=True)
    def check_content_provided(cls, v, values): #
        if not v and not values.get('topic_id') and not values.get('course_id'):
            raise ValueError('Either topic_id/course_id (for script generation) or raw_text_content must be provided.')
        return v

class GenerateVideoRequest(BaseModel): #
    user_id: str = Field(..., examples=["user_uuid_for_video_gen"])
    content_source: ContentSource
    instructor_character: InstructorChars # Using the Enum
    user_profile_snapshot: UserProfileSnapshotForTTV # For personalizing the script
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)
    preferred_video_length_minutes: str = Field("3-5", examples=["2-3", "4-6"], description="Target length for the video script.")
    preferred_attire_name: Optional[str] = Field(None, examples=["professional_blazer_blue_susan", "cozy_cardigan_green_trevor"])
    additional_instructions: Optional[str] = Field(None, examples=["Make the tone very encouraging.", "Focus on practical examples."])
    # InnovateAI Enhancement: Added for dynamic background selection
    background_theme_preference: Optional[str] = Field(None, examples=["tech_office", "calm_library", "dynamic_abstract"], description="Preference for video background theme if supported by avatar service.")


    @validator('language_code')
    def validate_language_code(cls, v): #
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language_code '{v}' received. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

# InnovateAI Enhancement: Added PREPARING_SCRIPT status
class VideoGenerationJobStatus(str, Enum): #
    PENDING = "pending"
    FETCHING_CONTENT = "fetching_content"
    PREPARING_SCRIPT = "preparing_script" # For SSML conversion and cue extraction
    GENERATING_SCRIPT = "generating_script"
    GENERATING_AUDIO = "generating_audio"
    SUBMITTING_TO_AVATAR_SERVICE = "submitting_to_avatar_service"
    RENDERING_AVATAR_VIDEO = "rendering_avatar_video"
    COMPLETED = "completed"
    FAILED = "failed"

class GenerateVideoInitialResponse(BaseModel): #
    job_id: str = Field(..., examples=[f"ttvjob_{uuid.uuid4()}"])
    status: VideoGenerationJobStatus = Field(VideoGenerationJobStatus.PENDING)
    message: str = Field("Video generation task accepted and queued.")
    estimated_completion_time_minutes: Optional[int] = Field(None, examples=[10, 20])

class VideoCallbackPayload(BaseModel): #
    job_id: str
    status: VideoGenerationJobStatus
    video_url: Optional[HttpUrl] = None
    thumbnail_url: Optional[HttpUrl] = None
    error_message: Optional[str] = None
    video_duration_seconds: Optional[float] = None
    script_generated_preview: Optional[str] = None
    character_used: Optional[str] = None
    attire_used: Optional[str] = None
    # InnovateAI Enhancement: Could add info about visual cues or background used if needed by Django
    # visual_cues_summary: Optional[List[str]] = None

# Local Enum for TTS Agent input type (to avoid direct cross-agent model import for now)
class TtsSynthesisInputType(str, Enum):
    TEXT = "text"
    SSML = "ssml"

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas Text-to-Video (TTV) Agent - Enhanced by InnovateAI",
    description="Orchestrates AI-driven script generation, SSML-enhanced TTS, and personalized avatar video synthesis.",
    version="0.3.0" # Incremented version
)

# --- In-memory "job store" (as per existing) ---
video_jobs: Dict[str, Dict[str, Any]] = {}

# --- Initialize Third-Party Avatar Client (as per existing, user to implement details) ---
if not THIRD_PARTY_AVATAR_API_KEY or not THIRD_PARTY_AVATAR_BASE_URL: #
    logger.warning("InnovateAI Note: THIRD_PARTY_AVATAR_API_KEY or THIRD_PARTY_AVATAR_BASE_URL not set. Avatar service calls will be mocked/fail. True 'Pixar-quality' depends on the chosen service's free tier capabilities!")
    avatar_service_client = ThirdPartyAvatarAPIClient(is_mock=True)
else:
    avatar_service_client = ThirdPartyAvatarAPIClient(
        api_key=THIRD_PARTY_AVATAR_API_KEY,
        base_url=THIRD_PARTY_AVATAR_BASE_URL
    )
    logger.info("InnovateAI Note: Avatar service client initialized. The magic of TTV awaits your chosen service's integration in avatar_api_client.py!")


# --- Helper Function to Update Job Status (as per existing) ---
async def update_job_status_and_notify(job_id: str, new_status: VideoGenerationJobStatus, **kwargs): #
    if job_id not in video_jobs:
        logger.error(f"Job ID {job_id} not found in store for status update to {new_status}.")
        return

    video_jobs[job_id]["status"] = new_status
    video_jobs[job_id].update(kwargs)
    logger.info(f"TTV Job {job_id}: Status updated to {new_status}. Details: {kwargs}")

    if new_status in [VideoGenerationJobStatus.COMPLETED, VideoGenerationJobStatus.FAILED]:
        if not DJANGO_TTV_CALLBACK_URL:
            logger.warning(f"DJANGO_TTV_CALLBACK_URL not set. Cannot send callback for job {job_id}.")
            return

        callback_payload = VideoCallbackPayload(
            job_id=job_id,
            status=new_status,
            video_url=video_jobs[job_id].get("video_url"),
            thumbnail_url=video_jobs[job_id].get("thumbnail_url"),
            error_message=video_jobs[job_id].get("error_message"),
            video_duration_seconds=video_jobs[job_id].get("video_duration_seconds"),
            script_generated_preview=video_jobs[job_id].get("script_generated_preview"),
            character_used=video_jobs[job_id].get("character_used"),
            attire_used=video_jobs[job_id].get("attire_used")
        )
        try:
            # **** Placeholder for actual HTTP call to Django - Mugambi to implement ****
            logger.info(f"InnovateAI Placeholder: Would send callback to Django for job {job_id} with status {new_status}.")
            # async with httpx.AsyncClient(timeout=30.0) as client:
            #     response = await client.post(DJANGO_TTV_CALLBACK_URL, json=callback_payload.model_dump(exclude_none=True))
            #     response.raise_for_status()
            #     logger.info(f"TTV Job {job_id}: Callback successfully sent to Django. Response: {response.status_code}")
        except httpx.HTTPStatusError as e:
            logger.error(f"TTV Job {job_id}: Failed to send callback to Django. Status: {e.response.status_code}, Body: {e.response.text if e.response else 'N/A'}", exc_info=True)
        except Exception as e:
            logger.error(f"TTV Job {job_id}: Exception during Django callback. Error: {e}", exc_info=True)


# --- InnovateAI Helper Functions for Enhanced TTV ---

def convert_script_to_ssml(raw_script: str, language_code: str, learning_pace: Optional[str]) -> str:
    logger.debug(f"InnovateAI: Original script for SSML conversion (lang: {language_code}, pace: {learning_pace}): {raw_script[:250]}...")
    ssml = raw_script

    # InnovateAI Note: These regex replacements are simplified examples.
    # For complex nested tags or more nuanced SSML, a proper XML/HTML parser might be better.
    # The AI Tutor should be prompted to generate script with these semantic tags.

    # Emphasis (ensure compatibility with Google TTS SSML)
    ssml = re.sub(r'<emphasis level="strong">(.*?)</emphasis>', r'<emphasis level="strong">\1</emphasis>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<emphasis level="moderate">(.*?)</emphasis>', r'<emphasis level="moderate">\1</emphasis>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<emphasis>(.*?)</emphasis>', r'<emphasis>\1</emphasis>', ssml, flags=re.IGNORECASE) # Default emphasis

    # Pauses (dynamic based on learning_pace)
    pause_duration_ms_map = {"slow": "700ms", "normal": "400ms", "fast": "200ms"}
    default_pause_ms = "400ms"
    
    def get_pause_ms(strength: Optional[str] = "medium") -> str:
        base_ms_str = pause_duration_ms_map.get(learning_pace or "normal", default_pause_ms)
        base_ms = int(base_ms_str[:-2]) # Remove "ms" and convert to int
        if strength == "long":
            return f"{int(base_ms * 1.5)}ms"
        elif strength == "short":
            return f"{int(base_ms * 0.5)}ms"
        return f"{base_ms}ms" # Medium

    ssml = re.sub(r'<pause strength="medium" />', f'<break time="{get_pause_ms("medium")}"/>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<pause strength="long" />', f'<break time="{get_pause_ms("long")}"/>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<pause strength="short" />', f'<break time="{get_pause_ms("short")}"/>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<pause />', f'<break time="{get_pause_ms("medium")}"/>', ssml, flags=re.IGNORECASE) # Default pause

    # Remove TTV-specific tags not meant for TTS (visuals, difficulty, placeholders)
    # Ensure these regexes are robust enough for variations if any.
    tags_to_remove_for_tts = [
        "visual_aid_suggestion", "difficulty", "topic",
        "analogy", "example" # Remove placeholders; actual content should be in script
    ]
    for tag_name in tags_to_remove_for_tts:
        ssml = re.sub(rf'<{tag_name}[^>]*>.*?</{tag_name}>', '', ssml, flags=re.DOTALL | re.IGNORECASE) # With content
        ssml = re.sub(rf'<{tag_name}[^>]*/>', '', ssml, flags=re.IGNORECASE) # Self-closing

    # Clean up any remaining empty/placeholder tags that might have been missed if AI Tutor used them differently
    ssml = re.sub(r'<[^/>]+type="[^"]*placeholder[^"]*"[^/>]*/>', '', ssml)


    # Ensure the final output is wrapped in <speak> tags for SSML standard
    # Also, escape XML special characters in the text content if not already done
    # For now, assuming script from AI Tutor has handled basic text content within tags.
    processed_ssml = ssml.strip()
    if not processed_ssml.lower().startswith("<speak>"):
        processed_ssml = f"<speak>{processed_ssml}</speak>"
    if not processed_ssml.lower().endswith("</speak>"):
        # This case should ideally not happen if wrapping is done correctly
        if processed_ssml.lower().startswith("<speak>"): # Remove if already started with speak
             processed_ssml = processed_ssml[len("<speak>"):]
        processed_ssml = f"<speak>{processed_ssml}</speak>"


    logger.debug(f"InnovateAI: Processed SSML for TTS: {processed_ssml[:300]}...")
    return processed_ssml

def extract_visual_aid_cues(script_with_tags: str) -> List[Dict[str, str]]:
    cues = []
    # Regex for: <visual_aid_suggestion type="diagram_needed" description="Flowchart of process X" />
    # Making it more flexible for attribute order or optional attributes.
    pattern = r'<visual_aid_suggestion\s+(?:type="([^"]+)"\s*)?(?:description="([^"]+)"\s*)?.*?/>'
    for match in re.finditer(pattern, script_with_tags, flags=re.IGNORECASE):
        cue_type = match.group(1) or "unknown"
        description = match.group(2) or "No description"
        cues.append({"type": cue_type, "description": description})
    logger.info(f"InnovateAI: Extracted {len(cues)} visual aid cues from script.")
    return cues

def determine_attire_tags(request_data: GenerateVideoRequest) -> List[str]:
    tags = ["daily_professional"] # Default attire tag
    if request_data.preferred_attire_name: # If user specified an exact attire, CharacterManager will use it.
        return [] # No tags needed if specific attire is requested by name

    if request_data.additional_instructions:
        instructions_lower = request_data.additional_instructions.lower()
        if "formal presentation" in instructions_lower or "keynote" in instructions_lower:
            tags = ["formal", "presentation"]
        elif "casual tutorial" in instructions_lower or "friendly explainer" in instructions_lower:
            tags = ["casual", "tutorial", "friendly"]
        elif "coding session" in instructions_lower:
            tags = ["smart_casual", "tech"]
    # InnovateAI Future Idea: Infer attire from course_id/topic_id metadata if available
    # e.g., if course is "Business Finance 101", add "business_formal" tag.
    logger.info(f"InnovateAI: Determined attire tags for character: {tags}")
    return tags

def determine_background_settings(request_data: GenerateVideoRequest, visual_cues: List[Dict]) -> Optional[Dict[str, Any]]:
    # InnovateAI Note: This function's output structure MUST match what your chosen
    # ThirdPartyAvatarAPIClient expects for its `background_settings` parameter.
    # This is a conceptual placeholder.
    if request_data.background_theme_preference:
        theme = request_data.background_theme_preference.lower()
        if theme == "tech_office":
            return {"type": "preset_scene_id", "id": "avatar_service_tech_office_scene_01"}
        elif theme == "calm_library":
            # Example: using a GCS URL for a background image
            return {"type": "image_url", "url": f"gs://{TTV_GCS_BUCKET_NAME}/backgrounds/library_calm.jpg"}
        elif theme == "dynamic_abstract":
            return {"type": "animated_loop_id", "id": "abstract_loop_blue_003"}
        elif theme.startswith("color:"): # e.g., color:#RRGGBB
            return {"type": "solid_color", "hex": theme.split(":")[1]}

    # Future idea: Use `visual_cues` to pick a more dynamic background if theme not set.
    # For example, if many cues are about "code" or "data", pick a techy background.
    # For now, this is a simple example.
    logger.info("InnovateAI: No specific background preference set or matched, avatar service default will be used.")
    return None


# --- Background Task for Video Generation Pipeline (Rewritten with Enhancements) ---
async def process_video_generation_task(job_id: str, request_data: GenerateVideoRequest): #
    logger.info(f"InnovateAI: Starting video generation task for job_id: {job_id}")
    await update_job_status_and_notify(job_id, VideoGenerationJobStatus.PENDING) # Initial status

    raw_script_from_tutor: Optional[str] = None
    processed_ssml_for_tts: Optional[str] = None
    generated_audio_gcs_url: Optional[str] = None
    tts_duration_seconds: Optional[float] = None
    visual_aid_cues: List[Dict[str,str]] = []

    try:
        # === Step 1: Generate Personalized Script via AI Tutor Agent ===
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.GENERATING_SCRIPT)
        logger.info(f"Job {job_id}: Requesting script from AI Tutor Agent. User: {request_data.user_id}, Topic: {request_data.content_source.topic_id or 'raw_text'}")
        
        # InnovateAI Note: The AI Tutor should be prompted to generate a script suitable for video,
        # incorporating placeholders or semantic cues that our NLP agent would have identified
        # (e.g., <analogy_placeholder>, <example_placeholder>, <visual_aid_cue>).
        # The AI Tutor would fill these placeholders based on user profile.
        script_query_for_tutor = ( #
            f"Generate a {request_data.preferred_video_length_minutes} minute video script for Uplas, explaining "
            f"{request_data.content_source.raw_text_content or ('topic ID: ' + (request_data.content_source.topic_id or 'the provided subject'))}. "
            f"The instructor persona is '{request_data.instructor_character.value}'. Personalize analogies and examples. "
            f"Include semantic tags like `<emphasis level=\"strong\">text</emphasis>`, `<pause strength=\"medium\" />`, and "
            f"`<visual_aid_suggestion type=\"diagram_needed\" description=\"A diagram of X\" />` where appropriate for video. "
            f"{request_data.additional_instructions or ''}"
        )

        tutor_payload = { #
            "user_id": request_data.user_id,
            "query_text": script_query_for_tutor,
            "user_profile_snapshot": request_data.user_profile_snapshot.model_dump(exclude_none=True),
            "language_code": request_data.language_code,
            "context": {
                "course_id": request_data.content_source.course_id,
                "topic_id": request_data.content_source.topic_id,
                # "current_topic_title": request_data.content_source.current_topic_title # If available
            },
            "max_tokens_response": 3072 # Allow ample space for script with potential tags
        }

        # **** Placeholder for actual HTTP call to AI Tutor - Mugambi to implement ****
        logger.info(f"InnovateAI Placeholder: Would call AI Tutor at {AI_TUTOR_AGENT_URL}/v1/ask-tutor")
        # async with httpx.AsyncClient(timeout=180.0) as client: # Increased timeout
        #     response = await client.post(f"{AI_TUTOR_AGENT_URL}/v1/ask-tutor", json=tutor_payload)
        #     response.raise_for_status()
        #     tutor_response_data = response.json()
        #     raw_script_from_tutor = tutor_response_data.get("answer_text")
        # Mocked response:
        raw_script_from_tutor = f"Hello {request_data.user_profile_snapshot.profession or 'learner'}! This is a script for {request_data.instructor_character.value} about {request_data.content_source.raw_text_content or request_data.content_source.topic_id}. <pause strength=\"medium\" /> <emphasis level=\"strong\">Pay attention!</emphasis> <visual_aid_suggestion type=\"chart\" description=\"Growth chart for Q1\"/>"
        # **** END Call ****

        if not raw_script_from_tutor or len(raw_script_from_tutor) < 20:
            raise ValueError(f"AI Tutor returned an insufficient script: '{raw_script_from_tutor}'")
        
        logger.info(f"Job {job_id}: Script generated (length: {len(raw_script_from_tutor)}). Preview: {raw_script_from_tutor[:100]}...")
        video_jobs[job_id]["script_generated_preview"] = raw_script_from_tutor[:500] # Store preview
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.PREPARING_SCRIPT)

        # === Step 1.5: InnovateAI - Preprocess Script to SSML & Extract Visual Cues ===
        logger.info(f"Job {job_id}: Converting script to SSML and extracting visual cues.")
        processed_ssml_for_tts = convert_script_to_ssml(
            raw_script_from_tutor,
            request_data.language_code,
            request_data.user_profile_snapshot.learning_pace_preference
        )
        visual_aid_cues = extract_visual_aid_cues(raw_script_from_tutor) # Use the original script with tags
        video_jobs[job_id]["visual_cues_identified"] = visual_aid_cues # Store for potential use or logging

        # === Step 2: Generate Audio using Enhanced TTS Agent (with SSML) ===
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.GENERATING_AUDIO)
        logger.info(f"Job {job_id}: Requesting audio from TTS Agent using SSML.")
        
        tts_voice_char_map = { #
            InstructorChars.SUSAN: "susan_us_standard", # Default to standard for cost; can add logic for WaveNet preference
            InstructorChars.UNCLE_TREVOR: "trevor_us_standard"
        }
        # InnovateAI Enhancement: Consider a flag in GenerateVideoRequest for "premium_audio"
        # if request_data.prefer_premium_audio:
        #    tts_voice_character = UPLAS_VOICE_CHARACTER_MAP.get(request_data.instructor_character.value + "_wavenet", default_voice)
        tts_voice_character = tts_voice_char_map.get(request_data.instructor_character, "default_en_us_standard")

        tts_payload = {
            "input_type": TtsSynthesisInputType.SSML.value, # Crucial: Specify SSML input
            "content_to_synthesize": processed_ssml_for_tts,
            "language_code": request_data.language_code,
            "voice_params": {
                "voice_character_name": tts_voice_character,
                # speaking_rate and pitch can be further controlled via SSML <prosody> tags if needed
            },
            "audio_config": {"audio_encoding": "MP3"} # Ensure this matches avatar service requirements
        }

        # **** Placeholder for actual HTTP call to TTS Agent - Mugambi to implement ****
        logger.info(f"InnovateAI Placeholder: Would call TTS Agent at {TTS_AGENT_URL}/v1/synthesize-speech")
        # async with httpx.AsyncClient(timeout=90.0) as client:
        #     response = await client.post(f"{TTS_AGENT_URL}/v1/synthesize-speech", json=tts_payload)
        #     response.raise_for_status()
        #     tts_response_data = response.json()
        #     generated_audio_gcs_url = tts_response_data.get("audio_url")
        #     tts_duration_seconds = tts_response_data.get("audio_duration_seconds")
        # Mocked response:
        generated_audio_gcs_url = f"gs://{TTV_GCS_BUCKET_NAME}/mocked_audio/{job_id}.mp3"
        tts_duration_seconds = float(len(processed_ssml_for_tts) / 15) # Very rough mock duration
        # **** END Call ****

        if not generated_audio_gcs_url:
            raise ValueError("TTS Agent did not return an audio URL.")
        logger.info(f"Job {job_id}: Audio generated successfully: {generated_audio_gcs_url}, Duration: {tts_duration_seconds}s")
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.SUBMITTING_TO_AVATAR_SERVICE, audio_gcs_url_temp=generated_audio_gcs_url)

        # === Step 3: Generate Avatar Video via Third-Party Service ===
        logger.info(f"Job {job_id}: Preparing data for Third-Party Avatar Service.")
        
        try:
            character_service_avatar_id = get_character_avatar_id_from_service(request_data.instructor_character.value) #
            
            attire_tags_for_selection = determine_attire_tags(request_data)
            chosen_attire_service_id = get_character_attire_id( #
                instructor_character_name=request_data.instructor_character.value,
                preferred_attire_name=request_data.preferred_attire_name,
                tags=attire_tags_for_selection
            )
            video_jobs[job_id]["attire_used"] = chosen_attire_service_id or "CHAR_DEFAULT_ATTIRE" # Store chosen attire
            video_jobs[job_id]["character_used"] = request_data.instructor_character.value # Store character


            background_settings_for_avatar = determine_background_settings(request_data, visual_aid_cues)
            if background_settings_for_avatar:
                 video_jobs[job_id]["background_details"] = background_settings_for_avatar # Log chosen background

        except CharacterConfigError as e: #
            logger.error(f"Job {job_id}: Character configuration error: {e}", exc_info=True)
            raise ValueError(f"Invalid character or attire configuration: {e}")

        logger.info(f"Job {job_id}: Submitting to Avatar Service. AvatarID: {character_service_avatar_id}, AttireID: {chosen_attire_service_id}, Audio: {generated_audio_gcs_url}")
        
        # **** Placeholder for actual call to Avatar Service Client - Mugambi to implement ****
        # This call will use the methods in animation_logic/avatar_api_client.py
        # Mugambi, this is where you'll integrate your chosen avatar service's API.
        logger.info(f"InnovateAI Placeholder: Would call avatar_service_client.submit_video_creation_job")
        # service_job_details = await avatar_service_client.submit_video_creation_job(
        #     service_avatar_id=character_service_avatar_id,
        #     audio_file_gcs_url=generated_audio_gcs_url,
        #     service_attire_id=chosen_attire_service_id,
        #     language_code=request_data.language_code, # Some services use this for lip sync accuracy
        #     background_settings=background_settings_for_avatar,
        #     # output_webhook_url=f"{YOUR_TTV_AGENT_URL}/v1/avatar-callback/{job_id}" # If service supports webhooks
        #     custom_metadata={"uplas_job_id": job_id, "user_id": request_data.user_id} # If service supports
        # )
        # Mocked response:
        mock_avatar_service_job_id = f"AVATAR_JOB_{uuid.uuid4().hex[:8]}"
        service_job_details = {"service_job_id": mock_avatar_service_job_id, "initial_status": "queued"}
        # **** END Call ****
        
        third_party_job_id = service_job_details.get("service_job_id")
        if not third_party_job_id:
            raise AvatarJobError("Avatar service did not return a job ID.")
        
        video_jobs[job_id]["third_party_avatar_job_id"] = third_party_job_id #
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.RENDERING_AVATAR_VIDEO, third_party_job_id=third_party_job_id)
        logger.info(f"Job {job_id}: Submitted to avatar service. Service Job ID: {third_party_job_id}")

        # === Step 4: Poll for Avatar Video Completion (Robust Polling) ===
        #
        max_polling_attempts = int(os.getenv("AVATAR_MAX_POLL_ATTEMPTS", 60)) # e.g., 30 mins if 30s interval
        poll_interval_seconds = int(os.getenv("AVATAR_POLL_INTERVAL_SECONDS", 30))
        initial_poll_delay_seconds = int(os.getenv("AVATAR_INITIAL_POLL_DELAY", 10)) # Initial delay before first poll

        await asyncio.sleep(initial_poll_delay_seconds) # Wait a bit before first poll

        final_video_url: Optional[str] = None
        final_thumbnail_url: Optional[str] = None
        video_duration: Optional[float] = tts_duration_seconds # Start with TTS duration as estimate

        for attempt in range(max_polling_attempts):
            logger.info(f"Job {job_id}: Polling avatar service for job {third_party_job_id}, attempt {attempt + 1}/{max_polling_attempts}")
            
            # **** Placeholder for actual call to Avatar Service Client - Mugambi to implement ****
            logger.info(f"InnovateAI Placeholder: Would call avatar_service_client.poll_video_job_status for {third_party_job_id}")
            # status_response = await avatar_service_client.poll_video_job_status(third_party_job_id)
            # Mocked polling response behavior:
            mock_progress = min(100, int(((attempt + 1) / (max_polling_attempts / 3.0)) * 100) ) # Simulate progress
            if attempt < (max_polling_attempts / 3): # Simulate processing
                status_response = {"status": "processing", "progressPercent": mock_progress}
            elif attempt < (max_polling_attempts - 5) : # Simulate rendering
                 status_response = {"status": "rendering", "progressPercent": mock_progress}
            else: # Simulate completion for last few attempts
                status_response = {
                    "status": "completed",
                    "videoUrl": f"https://mock-avatar-service.com/videos/{job_id}.mp4",
                    "thumbnailUrl": f"https://mock-avatar-service.com/videos/thumbs/{job_id}.jpg",
                    "durationSeconds": (tts_duration_seconds or 120) * random.uniform(0.9, 1.1) # Mock slight variation
                }
            # **** END Call ****
            
            current_service_status = status_response.get("status", "unknown").lower()
            service_progress = status_response.get("progressPercent")

            if current_service_status == "completed":
                final_video_url = status_response.get("videoUrl")
                final_thumbnail_url = status_response.get("thumbnailUrl")
                video_duration = status_response.get("durationSeconds", video_duration)
                if not final_video_url:
                    logger.error(f"Job {job_id}: Avatar service reported 'completed' but no video URL provided. Response: {status_response}")
                    raise AvatarJobError("Avatar service reported 'completed' but no video URL provided.")
                logger.info(f"Job {job_id}: Avatar video completed! URL: {final_video_url}")
                break
            elif current_service_status == "failed":
                error_msg = status_response.get("errorMessage", "Avatar generation failed with no specific message from service.")
                logger.error(f"Job {job_id}: Avatar service reported failure for {third_party_job_id}. Error: {error_msg}. Response: {status_response}")
                raise AvatarJobError(f"Avatar generation failed: {error_msg}")
            elif current_service_status in ["processing", "queued", "rendering", "submitted", "pending"]:
                logger.info(f"Job {job_id}: Avatar video status '{current_service_status}' (Progress: {service_progress}%). Polling again in {poll_interval_seconds}s...")
                await update_job_status_and_notify(job_id, VideoGenerationJobStatus.RENDERING_AVATAR_VIDEO, service_status_detail=f"{current_service_status} ({service_progress or 'N/A'}%)")
                await asyncio.sleep(poll_interval_seconds) # Standard polling interval
            else:
                logger.warning(f"Job {job_id}: Unknown status '{current_service_status}' from avatar service for {third_party_job_id}. Response: {status_response}. Will retry.")
                await asyncio.sleep(poll_interval_seconds * 2) # Longer wait for unknown status before retry
        else: # Loop finished without break (max_polling_attempts reached)
            logger.error(f"Job {job_id}: Timeout after {max_polling_attempts} attempts waiting for avatar video generation.")
            raise TimeoutError(f"Timeout waiting for avatar video generation for service job ID {third_party_job_id}.")

        # === Step 5: Finalize and Notify Django ===
        #
        # InnovateAI Note: Consider if final video/thumbnail URLs from avatar service need to be copied to your TTV_GCS_BUCKET_NAME
        # for long-term storage and CDN delivery, or if direct links from avatar service are acceptable (check their retention policies).
        logger.info(f"Job {job_id}: Finalizing job. Video URL: {final_video_url}")
        await update_job_status_and_notify(
            job_id,
            VideoGenerationJobStatus.COMPLETED,
            video_url=final_video_url,
            thumbnail_url=final_thumbnail_url,
            video_duration_seconds=round(video_duration, 2) if video_duration else None
            # character_used and attire_used are already set in video_jobs[job_id]
        )

    except ValueError as ve: # Catch our own validation errors (e.g. empty script)
        logger.error(f"TTV Job {job_id}: Validation error during processing. Error: {ve}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=f"Input validation error: {str(ve)}")
    except httpx.HTTPStatusError as hse: # Errors calling other internal agents
        logger.error(f"TTV Job {job_id}: HTTP error calling another service. Error: {hse.request.url} - {hse.response.status_code} - {hse.response.text}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=f"Dependency service error: Failed to call {hse.request.url}. Status: {hse.response.status_code}")
    except AvatarJobError as ae: # Errors specific to avatar service interaction
        logger.error(f"TTV Job {job_id}: Avatar service error. Error: {ae}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=f"Avatar service error: {str(ae)}")
    except TimeoutError as te: # Polling timeout
        logger.error(f"TTV Job {job_id}: Timeout error. Error: {te}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=str(te))
    except Exception as e: # Catch-all for unexpected errors
        logger.error(f"TTV Job {job_id}: Unhandled exception during video generation. Error: {e}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=f"An unexpected error occurred: {str(e)}")


# --- API Endpoints (Structure largely as per existing) ---
@app.post("/v1/generate-video", response_model=GenerateVideoInitialResponse, status_code=status.HTTP_202_ACCEPTED) #
async def generate_video_endpoint(request_data: GenerateVideoRequest, background_tasks: BackgroundTasks):
    if not all([GCP_PROJECT_ID, TTV_GCS_BUCKET_NAME, AI_TUTOR_AGENT_URL, TTS_AGENT_URL, DJANGO_TTV_CALLBACK_URL, THIRD_PARTY_AVATAR_API_KEY, THIRD_PARTY_AVATAR_BASE_URL]):
        # Check if mock mode is allowed despite missing full config
        if not (os.getenv("ALLOW_MOCK_MODE_IF_UNCONFIGURED", "false").lower() == "true" and avatar_service_client.is_mock):
             logger.error("TTV service critical configurations missing and mock mode not explicitly allowed.")
             raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="TTV service is not properly configured. Critical environment variables missing.")
        logger.warning("TTV service running with some missing configurations, relying on mock modes or defaults.")


    job_id = f"ttvjob_{uuid.uuid4()}"
    video_jobs[job_id] = { #
        "status": VideoGenerationJobStatus.PENDING,
        "requested_at": time.time(),
        "user_id": request_data.user_id,
        "instructor_character_requested": request_data.instructor_character.value,
        "language_requested": request_data.language_code,
        "error_message": None,
        "video_url": None,
        "script_generated_preview": None,
        "attire_used": None,
        "character_used": None,
        "third_party_avatar_job_id": None,
        "visual_cues_identified": [],
        "background_details": None
    }
    background_tasks.add_task(process_video_generation_task, job_id, request_data) #
    # Estimate can be more dynamic later based on requested length, etc.
    estimated_time_minutes = 5 + (int(request_data.preferred_video_length_minutes.split('-')[0]) * 3) # Rougher
    logger.info(f"Job {job_id} accepted for user {request_data.user_id}. Est. completion: {estimated_time_minutes} mins.")
    return GenerateVideoInitialResponse(job_id=job_id, status=VideoGenerationJobStatus.PENDING, estimated_completion_time_minutes=estimated_time_minutes, message="Video generation task accepted and queued by InnovateAI's TTV Orchestrator.")


@app.get("/v1/video-status/{job_id}", response_model=VideoCallbackPayload, summary="Get status of a video generation job") #
async def get_video_status_endpoint(job_id: str):
    job_info = video_jobs.get(job_id)
    if not job_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")
    
    # Ensure all fields for VideoCallbackPayload are present, even if None
    return VideoCallbackPayload(
        job_id=job_id,
        status=job_info.get("status", VideoGenerationJobStatus.FAILED), # Default to FAILED if status somehow missing
        video_url=job_info.get("video_url"),
        thumbnail_url=job_info.get("thumbnail_url"),
        error_message=job_info.get("error_message"),
        video_duration_seconds=job_info.get("video_duration_seconds"),
        script_generated_preview=job_info.get("script_generated_preview"),
        character_used=job_info.get("character_used"),
        attire_used=job_info.get("attire_used")
    )

@app.get("/health", status_code=status.HTTP_200_OK) #
async def health_check():
    # More comprehensive health check
    missing_configs = []
    if not AI_TUTOR_AGENT_URL: missing_configs.append("AI_TUTOR_AGENT_URL")
    if not TTS_AGENT_URL: missing_configs.append("TTS_AGENT_URL")
    if not DJANGO_TTV_CALLBACK_URL: missing_configs.append("DJANGO_TTV_CALLBACK_URL")
    if not avatar_service_client.is_mock and (not THIRD_PARTY_AVATAR_API_KEY or not THIRD_PARTY_AVATAR_BASE_URL):
        missing_configs.append("THIRD_PARTY_AVATAR_API_KEY/BASE_URL (for non-mock)")
    if not GCP_PROJECT_ID: missing_configs.append("GCP_PROJECT_ID")
    if not TTV_GCS_BUCKET_NAME: missing_configs.append("TTV_GCS_BUCKET_NAME")

    if missing_configs:
        return {"status": "unhealthy", "reason": f"Missing configurations: {', '.join(missing_configs)}", "service": "TTV_Agent"}
    return {"status": "healthy", "service": "TTV_Agent"}


if __name__ == "__main__": #
    import uvicorn
    logger.info("InnovateAI: Starting TTV Agent for local development...")
    # Example of how you might set ENVs for local run if not using .env or shell export
    # os.environ.setdefault("GCP_PROJECT_ID", "your-test-gcp-project")
    # ... etc. for other required ENVs for local testing.
    
    if not all([GCP_PROJECT_ID, TTV_GCS_BUCKET_NAME, AI_TUTOR_AGENT_URL, TTS_AGENT_URL, DJANGO_TTV_CALLBACK_URL]):
        print("WARNING: One or more critical environment variables for TTV agent are not set for full functionality.")
    if not avatar_service_client.is_mock and (not THIRD_PARTY_AVATAR_API_KEY or not THIRD_PARTY_AVATAR_BASE_URL):
        print("WARNING: Avatar service is NOT in mock mode, but API key/base URL might be missing.")

    port = int(os.getenv("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
