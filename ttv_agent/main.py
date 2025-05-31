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

# URLs for other Uplas AI Agents (these should be discoverable, e.g., via service discovery or env vars)
AI_TUTOR_AGENT_URL = os.getenv("AI_TUTOR_AGENT_URL", "http://localhost:8001") # Default for local dev if tutor runs on 8001
TTS_AGENT_URL = os.getenv("TTS_AGENT_URL", "http://localhost:8002") # Default for local dev if TTS runs on 8002

# API Key for the Third-Party Avatar Service (MUST BE SET IN ENV)
THIRD_PARTY_AVATAR_API_KEY = os.getenv("THIRD_PARTY_AVATAR_API_KEY")
THIRD_PARTY_AVATAR_BASE_URL = os.getenv("THIRD_PARTY_AVATAR_BASE_URL") # If needed by the client

# Supported languages - align with other agents
SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"]
DEFAULT_LANGUAGE = "en-US"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API Contract (Updated) ---

class UserProfileSnapshotForTTV(BaseModel): # Subset of UserProfile relevant to TTV script personalization
    industry: Optional[str] = Field(None, examples=["Healthcare"])
    profession: Optional[str] = Field(None, examples=["Nurse Practitioner"])
    country: Optional[str] = Field(None, examples=["Canada"])
    city: Optional[str] = Field(None, examples=["Toronto"])
    career_interest: Optional[str] = Field(None, examples=["AI in Medicine"])
    learning_goals: Optional[str] = Field(None, examples=["Understand how AI can improve diagnostics."])
    preferred_tutor_persona: Optional[str] = Field("Supportive and clear", examples=["Socratic", "Technical"])


class ContentSource(BaseModel):
    topic_id: Optional[str] = Field(None, examples=["topic_uuid_for_python_loops"])
    course_id: Optional[str] = Field(None, examples=["course_uuid_intro_python"])
    raw_text_content: Optional[str] = Field(None, examples=["Explain Python loops with an example for a beginner in finance."])
    # For TTV, we'll primarily rely on the AI Tutor to generate a script based on topic/course context.
    # If raw_text_content is provided, it might be a summary or key points to expand.

    @validator('raw_text_content', always=True)
    def check_content_provided(cls, v, values):
        # For TTV, we might primarily use topic_id/course_id to fetch structured content for script generation.
        # If raw_text_content is the only source, it needs to be substantial enough for a 3-5 min video script.
        if not v and not values.get('topic_id') and not values.get('course_id'):
            raise ValueError('Either topic_id/course_id (for script generation) or raw_text_content must be provided.')
        return v

class GenerateVideoRequest(BaseModel):
    user_id: str = Field(..., examples=["user_uuid_for_video_gen"])
    content_source: ContentSource
    instructor_character: InstructorChars # Using the Enum: UNCLE_TREVOR or SUSAN
    user_profile_snapshot: UserProfileSnapshotForTTV # For personalizing the script
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)
    preferred_video_length_minutes: str = Field("3-5", examples=["2-3", "4-6"], description="Target length for the video script.")
    # Optional: allow user to specify attire or let the system choose based on context/day
    preferred_attire_name: Optional[str] = Field(None, examples=["professional_blazer_blue_susan", "cozy_cardigan_green_trevor"])
    # Optional: specific instructions for the video, e.g., tone, style
    additional_instructions: Optional[str] = Field(None, examples=["Make the tone very encouraging.", "Focus on practical examples."])

    @validator('language_code')
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language_code '{v}' received. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

class VideoGenerationJobStatus(str, Enum):
    PENDING = "pending"
    FETCHING_CONTENT = "fetching_content" # If fetching from Django
    GENERATING_SCRIPT = "generating_script"
    GENERATING_AUDIO = "generating_audio"
    SUBMITTING_TO_AVATAR_SERVICE = "submitting_to_avatar_service"
    RENDERING_AVATAR_VIDEO = "rendering_avatar_video" # Status from third-party service
    COMPLETED = "completed"
    FAILED = "failed"

class GenerateVideoInitialResponse(BaseModel):
    job_id: str = Field(..., examples=[f"ttvjob_{uuid.uuid4()}"])
    status: VideoGenerationJobStatus = Field(VideoGenerationJobStatus.PENDING)
    message: str = Field("Video generation task accepted and queued.")
    estimated_completion_time_minutes: Optional[int] = Field(None, examples=[10, 20]) # Rough estimate

class VideoCallbackPayload(BaseModel): # For when video is ready (TTV agent calls Django)
    job_id: str
    status: VideoGenerationJobStatus
    video_url: Optional[HttpUrl] = None
    thumbnail_url: Optional[HttpUrl] = None
    error_message: Optional[str] = None
    video_duration_seconds: Optional[float] = None
    script_generated_preview: Optional[str] = None # First few lines of the script
    character_used: Optional[str] = None
    attire_used: Optional[str] = None

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas Text-to-Video (TTV) Agent (Real Integration Placeholder)",
    description="Orchestrates script generation, TTS, and avatar video synthesis.",
    version="0.2.0"
)

# --- In-memory "job store" for this agent ---
# In production, use Redis or a database for persistence.
video_jobs: Dict[str, Dict[str, Any]] = {}

# --- Initialize Third-Party Avatar Client ---
# This client needs to be implemented based on the chosen avatar service.
# It's initialized here to be used by the background task.
if not THIRD_PARTY_AVATAR_API_KEY or not THIRD_PARTY_AVATAR_BASE_URL:
    logger.warning("THIRD_PARTY_AVATAR_API_KEY or THIRD_PARTY_AVATAR_BASE_URL not set. Avatar service calls will be mocked/fail.")
    # Fallback to a mock client if not configured, or raise an error
    avatar_service_client = ThirdPartyAvatarAPIClient(is_mock=True) # Assuming client can run in mock mode
else:
    avatar_service_client = ThirdPartyAvatarAPIClient(
        api_key=THIRD_PARTY_AVATAR_API_KEY,
        base_url=THIRD_PARTY_AVATAR_BASE_URL
    )


# --- Helper Function to Update Job Status (and potentially send to Django) ---
async def update_job_status_and_notify(job_id: str, new_status: VideoGenerationJobStatus, **kwargs):
    if job_id not in video_jobs:
        logger.error(f"Job ID {job_id} not found in store for status update to {new_status}.")
        return

    video_jobs[job_id]["status"] = new_status
    video_jobs[job_id].update(kwargs) # Update with any additional info (video_url, error_message, etc.)
    logger.info(f"TTV Job {job_id}: Status updated to {new_status}. Details: {kwargs}")

    # If terminal status (COMPLETED or FAILED), send callback to Django
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
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(DJANGO_TTV_CALLBACK_URL, json=callback_payload.model_dump(exclude_none=True))
                response.raise_for_status()
                logger.info(f"TTV Job {job_id}: Callback successfully sent to Django. Response: {response.status_code}")
        except httpx.HTTPStatusError as e:
            logger.error(f"TTV Job {job_id}: Failed to send callback to Django. Status: {e.response.status_code}, Body: {e.response.text}", exc_info=True)
        except Exception as e:
            logger.error(f"TTV Job {job_id}: Exception during Django callback. Error: {e}", exc_info=True)


# --- Background Task for Video Generation Pipeline ---
async def process_video_generation_task(job_id: str, request_data: GenerateVideoRequest):
    """
    The actual video generation pipeline.
    1. Generate Script (via AI Tutor Agent)
    2. Generate Audio (via TTS Agent)
    3. Generate Avatar Video (via Third-Party Avatar Service)
    4. Store video and notify Django.
    """
    await update_job_status_and_notify(job_id, VideoGenerationJobStatus.GENERATING_SCRIPT)
    script_text: Optional[str] = None
    generated_audio_gcs_url: Optional[str] = None # URL from TTS agent, ideally already in GCS
    tts_duration_seconds: Optional[float] = None

    try:
        # === Step 1: Generate Script using AI Tutor Agent ===
        # The AI Tutor agent is better suited for generating personalized, instructional text.
        # We'll ask it for a script based on the context.
        logger.info(f"Job {job_id}: Requesting script from AI Tutor Agent for topic: {request_data.content_source.topic_id or 'raw_text'}")
        
        # Construct a query for the AI Tutor to generate a video script
        script_query = (
            f"Generate a {request_data.preferred_video_length_minutes} minute video script explaining "
            f"{request_data.content_source.raw_text_content or ('the topic: ' + (request_data.content_source.current_topic_title or request_data.content_source.topic_id or 'the provided subject'))}. "
            f"The script is for instructor '{request_data.instructor_character.value}'. "
            f"Incorporate analogies and examples relevant to the user. "
            f"{request_data.additional_instructions or ''} "
            "The script should be engaging and easy to follow. Structure it with scene descriptions or cues if possible, e.g., [VISUAL: ...], [SOUND: ...]."
        )

        tutor_payload = {
            "user_id": request_data.user_id,
            "query_text": script_query,
            "user_profile_snapshot": request_data.user_profile_snapshot.model_dump(),
            "language_code": request_data.language_code,
            "context": { # Pass context to AI Tutor
                "course_id": request_data.content_source.course_id,
                "topic_id": request_data.content_source.topic_id,
                # "current_topic_title": request_data.content_source.current_topic_title # If available
            },
            "max_tokens_response": 2048 # Allow for longer script
        }
        async with httpx.AsyncClient(timeout=120.0) as client: # Increased timeout for LLM script gen
            response = await client.post(f"{AI_TUTOR_AGENT_URL}/v1/ask-tutor", json=tutor_payload)
            response.raise_for_status()
            tutor_response_data = response.json()
            script_text = tutor_response_data.get("answer_text")

        if not script_text or len(script_text) < 50: # Basic check for meaningful script
            raise ValueError(f"AI Tutor returned an insufficient script: '{script_text}'")
        
        logger.info(f"Job {job_id}: Script generated successfully (length: {len(script_text)}). Preview: {script_text[:200]}...")
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.GENERATING_AUDIO, script_generated_preview=script_text[:500])

        # === Step 2: Generate Audio using TTS Agent ===
        logger.info(f"Job {job_id}: Requesting audio from TTS Agent for script.")
        
        # Determine voice character for TTS based on TTV instructor
        # This mapping should be robust. For now, a simple assumption.
        tts_voice_char_map = {
            InstructorChars.SUSAN: "susan_pro_voice", # Defined in your TTS Agent's UPLAS_VOICE_CHARACTER_MAP
            InstructorChars.UNCLE_TREVOR: "trevor_wise_voice"
        }
        tts_voice_character = tts_voice_char_map.get(request_data.instructor_character, "default_en_us_female")

        tts_payload = {
            "text_to_speak": script_text,
            "language_code": request_data.language_code, # TTS agent will use this
            "voice_params": {"voice_character_name": tts_voice_character},
            "audio_config": {"audio_encoding": "MP3"} # Avatar services often prefer MP3 or WAV
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{TTS_AGENT_URL}/v1/synthesize-speech", json=tts_payload)
            response.raise_for_status()
            tts_response_data = response.json()
            generated_audio_gcs_url = tts_response_data.get("audio_url")
            tts_duration_seconds = tts_response_data.get("audio_duration_seconds")

        if not generated_audio_gcs_url:
            raise ValueError("TTS Agent did not return an audio URL.")
        
        logger.info(f"Job {job_id}: Audio generated successfully: {generated_audio_gcs_url}")
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.SUBMITTING_TO_AVATAR_SERVICE, audio_gcs_url_temp=generated_audio_gcs_url)

        # === Step 3: Generate Avatar Video ===
        logger.info(f"Job {job_id}: Submitting job to Third-Party Avatar Service.")
        
        # Get character and attire details using CharacterManager
        try:
            character_service_avatar_id = get_character_avatar_id_from_service(request_data.instructor_character.value)
            
            # Attire selection logic (can be more sophisticated)
            attire_tags = ["daily_professional"] # Default tags
            if "formal" in (request_data.additional_instructions or "").lower():
                attire_tags = ["formal"]
            
            chosen_attire_service_id = get_character_attire_id(
                instructor_character_name=request_data.instructor_character.value,
                preferred_attire_name=request_data.preferred_attire_name,
                tags=attire_tags
            )
            # Store the name of the attire for the callback
            # This requires mapping service_attire_id back to a name, or storing the name initially.
            # For simplicity, we'll just use the service ID for now in the log.
            video_jobs[job_id]["attire_used"] = chosen_attire_service_id or "default"

        except CharacterConfigError as e:
            logger.error(f"Job {job_id}: Character configuration error: {e}", exc_info=True)
            raise ValueError(f"Invalid character or attire configuration: {e}")

        # Submit to the third-party avatar service
        service_job_details = await avatar_service_client.submit_video_creation_job(
            service_avatar_id=character_service_avatar_id,
            audio_file_gcs_url=generated_audio_gcs_url, # Pass the GCS URL of the audio
            # language_code=request_data.language_code, # Service might need this for lip-sync accuracy if not inferred
            service_attire_id=chosen_attire_service_id,
            # background_settings, output_webhook_url etc. can be added if service supports
        )
        
        third_party_job_id = service_job_details.get("service_job_id")
        if not third_party_job_id:
            raise AvatarJobError("Avatar service did not return a job ID.")
        
        video_jobs[job_id]["third_party_avatar_job_id"] = third_party_job_id
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.RENDERING_AVATAR_VIDEO, third_party_job_id=third_party_job_id)
        logger.info(f"Job {job_id}: Submitted to avatar service. Service Job ID: {third_party_job_id}")

        # === Step 4: Poll for Avatar Video Completion (or use webhook if service supports it) ===
        # This is a simplified polling loop. Production systems might use webhooks or more robust polling.
        max_polling_attempts = 60 # e.g., 30 minutes if polling every 30s
        poll_interval_seconds = 30 
        final_video_url: Optional[str] = None
        final_thumbnail_url: Optional[str] = None
        video_duration: Optional[float] = tts_duration_seconds # Use TTS duration as an estimate

        for attempt in range(max_polling_attempts):
            await asyncio.sleep(poll_interval_seconds) # Use asyncio.sleep for async background task
            logger.info(f"Job {job_id}: Polling avatar service for job {third_party_job_id}, attempt {attempt + 1}")
            status_response = await avatar_service_client.poll_video_job_status(third_party_job_id)
            
            current_service_status = status_response.get("status")
            if current_service_status == "completed":
                final_video_url = status_response.get("videoUrl")
                final_thumbnail_url = status_response.get("thumbnailUrl")
                video_duration = status_response.get("durationSeconds", tts_duration_seconds) # Prefer service duration
                if not final_video_url:
                    raise AvatarJobError("Avatar service reported 'completed' but no video URL provided.")
                logger.info(f"Job {job_id}: Avatar video completed. URL: {final_video_url}")
                break
            elif current_service_status == "failed":
                error_msg = status_response.get("errorMessage", "Avatar generation failed with no specific message.")
                raise AvatarJobError(f"Avatar generation failed: {error_msg}")
            elif current_service_status in ["processing", "queued", "rendering"]:
                logger.info(f"Job {job_id}: Avatar video still '{current_service_status}'. Polling again...")
                # Update job status if you want to reflect these intermediate avatar service statuses
                await update_job_status_and_notify(job_id, VideoGenerationJobStatus.RENDERING_AVATAR_VIDEO, service_status_detail=current_service_status)
            else:
                logger.warning(f"Job {job_id}: Unknown status from avatar service: {current_service_status}")
        else: # Loop finished without break (max_polling_attempts reached)
            raise TimeoutError("Timeout waiting for avatar video generation to complete.")

        # === Step 5: Finalize and Notify ===
        # Optionally, copy the video from the third-party service's storage to your own GCS bucket
        # For now, we assume final_video_url is directly usable or points to a GCS location.
        
        await update_job_status_and_notify(
            job_id,
            VideoGenerationJobStatus.COMPLETED,
            video_url=final_video_url,
            thumbnail_url=final_thumbnail_url,
            video_duration_seconds=video_duration,
            character_used=request_data.instructor_character.value
        )

    except Exception as e:
        logger.error(f"TTV Job {job_id}: Failed. Error: {e}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=str(e))

# --- API Endpoints ---

@app.post("/v1/generate-video", response_model=GenerateVideoInitialResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_video_endpoint(request_data: GenerateVideoRequest, background_tasks: BackgroundTasks):
    """
    Accepts a request to generate an instructor video.
    Starts the generation process in the background and returns a job ID.
    """
    if not GCP_PROJECT_ID or not TTV_GCS_BUCKET_NAME: # Basic config check
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="TTV service is not properly configured (missing GCP Project ID or GCS Bucket).")

    job_id = f"ttvjob_{uuid.uuid4()}"
    video_jobs[job_id] = {
        "status": VideoGenerationJobStatus.PENDING,
        "requested_at": time.time(),
        "user_id": request_data.user_id,
        "instructor_character_requested": request_data.instructor_character.value,
        "language_requested": request_data.language_code,
        "error_message": None,
        "video_url": None
    }
    background_tasks.add_task(process_video_generation_task, job_id, request_data)
    estimated_time = 15 # Default rough estimate in minutes
    return GenerateVideoInitialResponse(job_id=job_id, status=VideoGenerationJobStatus.PENDING, estimated_completion_time_minutes=estimated_time)


@app.get("/v1/video-status/{job_id}", response_model=VideoCallbackPayload, summary="Get status of a video generation job")
async def get_video_status_endpoint(job_id: str):
    """
    Allows polling for the status of a video generation job.
    """
    job_info = video_jobs.get(job_id)
    if not job_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")
    
    return VideoCallbackPayload(
        job_id=job_id,
        status=job_info.get("status", VideoGenerationJobStatus.FAILED),
        video_url=job_info.get("video_url"),
        thumbnail_url=job_info.get("thumbnail_url"),
        error_message=job_info.get("error_message"),
        video_duration_seconds=job_info.get("video_duration_seconds"),
        script_generated_preview=job_info.get("script_generated_preview"),
        character_used=job_info.get("character_used"),
        attire_used=job_info.get("attire_used")
    )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    # Basic check for critical configurations
    if not AI_TUTOR_AGENT_URL or not TTS_AGENT_URL:
        return {"status": "unhealthy", "reason": "Dependent agent URLs not configured.", "service": "TTV_Agent"}
    if not DJANGO_TTV_CALLBACK_URL:
        return {"status": "unhealthy", "reason": "Django callback URL not configured.", "service": "TTV_Agent"}
    if not avatar_service_client or (not avatar_service_client.is_mock and (not THIRD_PARTY_AVATAR_API_KEY or not THIRD_PARTY_AVATAR_BASE_URL)):
         return {"status": "unhealthy", "reason": "Third-party avatar service not configured.", "service": "TTV_Agent"}
    return {"status": "healthy", "service": "TTV_Agent"}

# For asyncio.sleep in background task
import asyncio

if __name__ == "__main__":
    import uvicorn
    # Ensure environment variables are set for local testing
    # e.g., GCP_PROJECT_ID, TTV_GCS_BUCKET_NAME, DJANGO_TTV_CALLBACK_URL,
    # AI_TUTOR_AGENT_URL, TTS_AGENT_URL,
    # THIRD_PARTY_AVATAR_API_KEY, THIRD_PARTY_AVATAR_BASE_URL
    
    # Example:
    # GCP_PROJECT_ID="your-gcp-project" TTV_GCS_BUCKET_NAME="your-ttv-bucket" \
    # DJANGO_TTV_CALLBACK_URL="http://localhost:8000/api/internal/ttv-callback/" \
    # AI_TUTOR_AGENT_URL="http://localhost:8001" TTS_AGENT_URL="http://localhost:8002" \
    # THIRD_PARTY_AVATAR_API_KEY="your_avatar_service_key" \
    # THIRD_PARTY_AVATAR_BASE_URL="https://api.avatarservice.com/v1" \
    # uvicorn main:app --reload --port 8003

    if not all([GCP_PROJECT_ID, TTV_GCS_BUCKET_NAME, DJANGO_TTV_CALLBACK_URL, AI_TUTOR_AGENT_URL, TTS_AGENT_URL]):
        print("Warning: One or more critical environment variables for TTV agent are not set.")
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8003)))

