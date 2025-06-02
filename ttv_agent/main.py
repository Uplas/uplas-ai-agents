# uplas-ai-agents/ttv_agent/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import os
import uuid
import time
import httpx 
import logging
import random 
import re 
import asyncio 
from io import BytesIO

# GCP Clients (for Slideshow MVP)
from google.cloud import transcoder_v1 # For Video Stitching via Transcoder API
from google.cloud.transcoder_v1.services.transcoder_service import TranscoderServiceClient
from google.protobuf import json_format # For constructing Transcoder job config

# Image generation for slides (Pillow)
from PIL import Image, ImageDraw, ImageFont

from .animation_logic.character_manager import (
    InstructorChars, 
    get_character_config,
    get_avatar_service_id as get_character_avatar_id_from_service, 
    get_voice_settings as get_character_voice_settings, 
    get_attire_id as get_character_attire_id, 
    CharacterConfigError
)
from .animation_logic.avatar_api_client import ThirdPartyAvatarAPIClient, AvatarJobError


# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") # Transcoder API needs a location
TTV_GCS_BUCKET_NAME = os.getenv("TTV_GCS_BUCKET_NAME") 
DJANGO_TTV_CALLBACK_URL = os.getenv("DJANGO_TTV_CALLBACK_URL") 

AI_TUTOR_AGENT_URL = os.getenv("AI_TUTOR_AGENT_URL", "http://localhost:8001") 
TTS_AGENT_URL = os.getenv("TTS_AGENT_URL", "http://localhost:8002") 

THIRD_PARTY_AVATAR_API_KEY = os.getenv("THIRD_PARTY_AVATAR_API_KEY")
THIRD_PARTY_AVATAR_BASE_URL = os.getenv("THIRD_PARTY_AVATAR_BASE_URL") 

# NovaSpark: Configuration for TTV Mode (Task 1 Option B)
# 'FULL_AVATAR' (default) or 'SLIDESHOW_MVP'
TTV_OPERATING_MODE = os.getenv("TTV_OPERATING_MODE", "FULL_AVATAR").upper()

# NovaSpark: Placeholder for Stock Photo API (Slideshow MVP)
STOCK_PHOTO_API_KEY_ENV = "PEXELS_API_KEY" # Example, if using Pexels
STOCK_PHOTO_API_URL_ENV = "PEXELS_API_URL" # Example

SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"]
DEFAULT_LANGUAGE = "en-US"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class UserProfileSnapshotForTTV(BaseModel): 
    industry: Optional[str] = Field(None, examples=["Healthcare", "Technology", "Education"])
    profession: Optional[str] = Field(None, examples=["Nurse Practitioner", "Software Developer", "Teacher"])
    country: Optional[str] = Field(None, examples=["Canada"])
    city: Optional[str] = Field(None, examples=["Toronto"])
    career_interest: Optional[str] = Field(None, examples=["AI in Medicine", "Data Science education"])
    learning_goals: Optional[str] = Field(None, examples=["Understand how AI can improve diagnostics.", "Create engaging explainers."])
    # NovaSpark Task 2: Add fields that can directly influence visual choices
    preferred_visual_style: Optional[str] = Field(None, examples=["modern_clean", "playful_illustrative", "corporate_formal"])
    favorite_colors: Optional[List[str]] = Field(default_factory=list, examples=[["blue", "green"], ["#FFD700"]])
    preferred_tutor_persona: Optional[str] = Field("Supportive and clear", examples=["Socratic", "Technical"]) # Used by AI Tutor for script
    learning_pace_preference: Optional[str] = Field("normal", examples=["slow", "normal", "fast"])


class ContentSource(BaseModel): 
    topic_id: Optional[str] = Field(None, examples=["topic_uuid_for_python_loops"])
    course_id: Optional[str] = Field(None, examples=["course_uuid_intro_python"])
    raw_text_content: Optional[str] = Field(None, examples=["Explain Python loops with an example for a beginner in finance."])
    @validator('raw_text_content', always=True)
    def check_content_provided(cls, v, values): 
        if not v and not values.get('topic_id') and not values.get('course_id'):
            raise ValueError('Either topic_id/course_id (for script generation) or raw_text_content must be provided.')
        return v

class GenerateVideoRequest(BaseModel): 
    user_id: str = Field(..., examples=["user_uuid_for_video_gen"])
    content_source: ContentSource
    instructor_character: InstructorChars 
    user_profile_snapshot: UserProfileSnapshotForTTV # For personalizing script & visuals
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)
    preferred_video_length_minutes: str = Field("3-5", examples=["2-3", "4-6"])
    preferred_attire_name: Optional[str] = Field(None, examples=["professional_blazer_blue_susan", "cozy_cardigan_green_trevor"])
    additional_instructions: Optional[str] = Field(None, examples=["Make the tone very encouraging.", "Focus on practical examples."])
    background_theme_preference: Optional[str] = Field(None, examples=["tech_office", "calm_library", "dynamic_abstract", "user_color_preference"])

    @validator('language_code')
    def validate_language_code(cls, v): 
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"NovaSpark Warning: Unsupported language_code '{v}' received. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

class VideoGenerationJobStatus(str, Enum): 
    PENDING = "pending"
    FETCHING_CONTENT = "fetching_content" # Script from source or AI Tutor
    PREPARING_SCRIPT = "preparing_script" # SSML conversion, cue extraction
    GENERATING_AUDIO = "generating_audio" # TTS call
    # Full Avatar Mode
    SUBMITTING_TO_AVATAR_SERVICE = "submitting_to_avatar_service"
    RENDERING_AVATAR_VIDEO = "rendering_avatar_video"
    # Slideshow MVP Mode (Task 1)
    GENERATING_SLIDES = "generating_slides"
    SUBMITTING_TO_TRANSCODER = "submitting_to_transcoder"
    RENDERING_TRANSCODED_VIDEO = "rendering_transcoded_video"
    COMPLETED = "completed"
    FAILED = "failed"

class GenerateVideoInitialResponse(BaseModel): 
    job_id: str = Field(..., examples=[f"ttvjob_{uuid.uuid4()}"])
    status: VideoGenerationJobStatus = Field(VideoGenerationJobStatus.PENDING)
    message: str = Field("Video generation task accepted and queued.")
    estimated_completion_time_minutes: Optional[int] = Field(None, examples=[10, 20])

class VideoCallbackPayload(BaseModel): 
    job_id: str
    status: VideoGenerationJobStatus
    video_url: Optional[HttpUrl] = None
    thumbnail_url: Optional[HttpUrl] = None
    error_message: Optional[str] = None
    video_duration_seconds: Optional[float] = None
    script_generated_preview: Optional[str] = None
    character_used: Optional[str] = None # Name of Uplas character
    attire_used: Optional[str] = None # Name of Uplas attire
    video_generation_mode: Optional[str] = Field(None, description="Indicates if 'FULL_AVATAR' or 'SLIDESHOW_MVP' was used.")


class TtsSynthesisInputType(str, Enum): TEXT = "text"; SSML = "ssml"

app = FastAPI(
    title="Uplas Text-to-Video (TTV) Agent - NovaSpark Edition",
    description="Orchestrates AI-driven script generation, TTS, and personalized video synthesis (Avatar or Slideshow MVP).",
    version="0.4.0" # Incremented for Task 1 & 2 TTV enhancements
)

video_jobs: Dict[str, Dict[str, Any]] = {}

# Initialize Avatar Client (for FULL_AVATAR mode)
if TTV_OPERATING_MODE == "FULL_AVATAR":
    if not THIRD_PARTY_AVATAR_API_KEY or not THIRD_PARTY_AVATAR_BASE_URL:
        logger.warning("NovaSpark TTV Warning: FULL_AVATAR mode selected, but THIRD_PARTY_AVATAR_API_KEY or _BASE_URL not set. Avatar service calls will be mocked/fail.")
        avatar_service_client = ThirdPartyAvatarAPIClient(is_mock_override=True)
    else:
        avatar_service_client = ThirdPartyAvatarAPIClient(api_key=THIRD_PARTY_AVATAR_API_KEY, base_url=THIRD_PARTY_AVATAR_BASE_URL)
        logger.info("NovaSpark TTV: Avatar service client initialized for FULL_AVATAR mode.")
else: # SLIDESHOW_MVP or other
    avatar_service_client = None # Not needed for slideshow
    logger.info(f"NovaSpark TTV: Operating in {TTV_OPERATING_MODE} mode. Third-party avatar client not initialized.")

# Initialize GCP Transcoder Client (for SLIDESHOW_MVP mode)
if TTV_OPERATING_MODE == "SLIDESHOW_MVP":
    try:
        transcoder_client = TranscoderServiceClient()
        logger.info("NovaSpark TTV: GCP Transcoder client initialized for SLIDESHOW_MVP mode.")
    except Exception as e_transcoder_init:
        logger.error(f"NovaSpark TTV Critical: Failed to initialize GCP Transcoder client: {e_transcoder_init}", exc_info=True)
        transcoder_client = None # Agent might be unhealthy if this fails
else:
    transcoder_client = None


async def update_job_status_and_notify(job_id: str, new_status: VideoGenerationJobStatus, **kwargs): 
    # ... (update_job_status_and_notify from main (14).py - minor addition for video_generation_mode)
    if job_id not in video_jobs: logger.error(f"Job ID {job_id} not found for status update to {new_status}."); return
    video_jobs[job_id]["status"] = new_status
    video_jobs[job_id].update(kwargs)
    logger.info(f"TTV Job {job_id}: Status updated to {new_status}. Details: {kwargs}")

    if new_status in [VideoGenerationJobStatus.COMPLETED, VideoGenerationJobStatus.FAILED]:
        if not DJANGO_TTV_CALLBACK_URL: logger.warning(f"DJANGO_TTV_CALLBACK_URL not set. Cannot send callback for job {job_id}."); return
        payload_dict = {
            "job_id": job_id, "status": new_status.value,
            "video_url": video_jobs[job_id].get("video_url"), "thumbnail_url": video_jobs[job_id].get("thumbnail_url"),
            "error_message": video_jobs[job_id].get("error_message"), "video_duration_seconds": video_jobs[job_id].get("video_duration_seconds"),
            "script_generated_preview": video_jobs[job_id].get("script_generated_preview"), "character_used": video_jobs[job_id].get("character_used"),
            "attire_used": video_jobs[job_id].get("attire_used"), "video_generation_mode": video_jobs[job_id].get("video_generation_mode", TTV_OPERATING_MODE) # Add mode to callback
        }
        callback_payload = VideoCallbackPayload(**payload_dict)
        try:
            # NovaSpark: Added timeout and basic retry to this HTTP call
            async with httpx.AsyncClient(timeout=30.0, transport=httpx.AsyncHTTPTransport(retries=2)) as client:
                response = await client.post(DJANGO_TTV_CALLBACK_URL, json=callback_payload.model_dump(exclude_none=True))
                response.raise_for_status()
                logger.info(f"TTV Job {job_id}: Callback successfully sent to Django. Response: {response.status_code}")
        except Exception as e:
            logger.error(f"TTV Job {job_id}: Exception during Django callback. Error: {e}", exc_info=True)

# --- Helper Functions ---
# convert_script_to_ssml, extract_visual_aid_cues from main (14).py - assumed okay for now.
def convert_script_to_ssml(raw_script: str, language_code: str, learning_pace: Optional[str]) -> str:
    # ... (logic from main (14).py) ...
    logger.debug(f"NovaSpark SSML Conversion: Original: {raw_script[:100]}..., Pace: {learning_pace}")
    ssml = raw_script
    pause_duration_ms_map = {"slow": "700ms", "normal": "400ms", "fast": "200ms"}; default_pause_ms = "400ms"
    def get_pause_ms(strength: Optional[str] = "medium") -> str:
        base_ms = int(pause_duration_ms_map.get(learning_pace or "normal", default_pause_ms)[:-2])
        if strength == "long": return f"{int(base_ms * 1.5)}ms"
        if strength == "short": return f"{int(base_ms * 0.5)}ms"
        return f"{base_ms}ms"
    ssml = re.sub(r'<emphasis level="strong">(.*?)</emphasis>', r'<emphasis level="strong">\1</emphasis>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<emphasis level="moderate">(.*?)</emphasis>', r'<emphasis level="moderate">\1</emphasis>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<emphasis>(.*?)</emphasis>', r'<emphasis>\1</emphasis>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<pause strength="medium" />', f'<break time="{get_pause_ms("medium")}"/>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<pause strength="long" />', f'<break time="{get_pause_ms("long")}"/>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<pause strength="short" />', f'<break time="{get_pause_ms("short")}"/>', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<pause />', f'<break time="{get_pause_ms("medium")}"/>', ssml, flags=re.IGNORECASE)
    tags_to_remove_for_tts = ["visual_aid_suggestion", "difficulty", "topic", "analogy", "example"]
    for tag_name in tags_to_remove_for_tts:
        ssml = re.sub(rf'<{tag_name}[^>]*>.*?</{tag_name}>', '', ssml, flags=re.DOTALL | re.IGNORECASE)
        ssml = re.sub(rf'<{tag_name}[^>]*/>', '', ssml, flags=re.IGNORECASE)
    ssml = re.sub(r'<[^/>]+type="[^"]*placeholder[^"]*"[^/>]*/>', '', ssml)
    processed_ssml = ssml.strip()
    if not processed_ssml.lower().startswith("<speak>"): processed_ssml = f"<speak>{processed_ssml}"
    if not processed_ssml.lower().endswith("</speak>"): processed_ssml = f"{processed_ssml}</speak>"
    # A common issue: ensure <speak> is only at the very start and end if content was empty.
    if processed_ssml == "<speak></speak>" and raw_script.strip(): # if raw script had content but ssml is empty, preserve some raw
        logger.warning("NovaSpark SSML: SSML became empty after tag stripping, using raw script fragment for TTS.")
        return f"<speak>{raw_script[:4900]}</speak>" # Max 5000 chars for TTS
    if not processed_ssml.strip() or processed_ssml == "<speak></speak>": # if truly empty
        logger.warning("NovaSpark SSML: Script resulted in empty SSML. Returning empty speak tag.")
        return "<speak> </speak>" # TTS API might require non-empty content within speak
        
    logger.debug(f"NovaSpark SSML Conversion: Result: {processed_ssml[:100]}...")
    return processed_ssml


def extract_visual_aid_cues(script_with_tags: str) -> List[Dict[str, str]]:
    # ... (logic from main (14).py) ...
    cues = []
    pattern = r'<visual_aid_suggestion\s+(?:type="([^"]+)"\s*)?(?:description="([^"]+)"\s*)?(?:keywords="([^"]+)"\s*)?.*?/>' # Task 5: Added keywords
    for match in re.finditer(pattern, script_with_tags, flags=re.IGNORECASE):
        cues.append({
            "type": match.group(1) or "unknown", 
            "description": match.group(2) or "No description",
            "keywords": match.group(3) or "" # Task 5
        })
    logger.info(f"NovaSpark: Extracted {len(cues)} visual aid cues. First cue: {cues[0] if cues else 'N/A'}")
    return cues

# NovaSpark Enhanced: determine_attire_tags (Task 2 Personalization)
def determine_attire_tags(request_data: GenerateVideoRequest) -> List[str]:
    tags = [] # Start with empty, more specific tags will be added
    profile = request_data.user_profile_snapshot

    if request_data.preferred_attire_name: return [] # Exact name overrides tags

    # 1. Infer from professional context
    if profile.profession:
        prof_lower = profile.profession.lower()
        if "doctor" in prof_lower or "nurse" in prof_lower or "physician" in prof_lower: tags.append("medical_professional")
        elif "engineer" in prof_lower or "developer" in prof_lower: tags.append("tech_professional"); tags.append("smart_casual")
        elif "teacher" in prof_lower or "educator" in prof_lower: tags.append("educator_style"); tags.append("approachable")
        elif "executive" in prof_lower or "manager" in prof_lower or "ceo" in prof_lower: tags.append("business_formal")
        elif "artist" in prof_lower or "designer" in prof_lower: tags.append("creative_professional"); tags.append("modern_casual")
    
    if profile.industry:
        ind_lower = profile.industry.lower()
        if "finance" in ind_lower or "banking" in ind_lower: tags.append("business_formal")
        elif "technology" in ind_lower or "software" in ind_lower: tags.append("tech_look")
        elif "education" in ind_lower: tags.append("academic_style")

    # 2. Infer from additional instructions (existing logic)
    if request_data.additional_instructions:
        instr_lower = request_data.additional_instructions.lower()
        if "formal presentation" in instr_lower or "keynote" in instr_lower: tags.extend(["formal", "presentation"])
        elif "casual tutorial" in instr_lower or "friendly explainer" in instr_lower: tags.extend(["casual", "tutorial", "friendly"])
        elif "coding session" in instr_lower: tags.extend(["smart_casual", "tech_demo"])
        elif "storytelling" in instr_lower: tags.append("storyteller_vibe")

    # 3. Infer from learning goals or preferred persona
    if profile.learning_goals and "presenting" in profile.learning_goals.lower(): tags.append("presenter_style")
    if profile.preferred_tutor_persona:
        persona_lower = profile.preferred_tutor_persona.lower()
        if "socratic" in persona_lower or "mentor" in persona_lower: tags.append("thoughtful_mentor")
        elif "technical" in persona_lower: tags.append("expert_technical")

    if not tags: tags.append("daily_professional") # Fallback default
    
    unique_tags = list(set(tags)) # Remove duplicates
    logger.info(f"NovaSpark TTV Personalization: Determined attire tags for character: {unique_tags} based on profile and instructions.")
    return unique_tags

# NovaSpark Enhanced: determine_background_settings (Task 2 Personalization)
def determine_background_settings(request_data: GenerateVideoRequest, visual_cues: List[Dict]) -> Optional[Dict[str, Any]]:
    profile = request_data.user_profile_snapshot
    
    # Priority 1: Explicit user preference
    if request_data.background_theme_preference:
        theme_pref = request_data.background_theme_preference.lower()
        if theme_pref == "tech_office": return {"type": "preset_scene_id", "id": "avatar_service_tech_office_01"} # Placeholder ID
        elif theme_pref == "calm_library": return {"type": "image_url", "url": f"gs://{TTV_GCS_BUCKET_NAME}/backgrounds/library_calm_01.jpg"}
        elif theme_pref == "dynamic_abstract": return {"type": "animated_loop_id", "id": "abstract_loop_blue_tech_003"}
        elif theme_pref.startswith("user_color_preference") and profile.favorite_colors:
            return {"type": "solid_color", "hex": profile.favorite_colors[0]} # Use first favorite color
        elif theme_pref.startswith("color:"): return {"type": "solid_color", "hex": theme_pref.split(":")[1]}
        # Add more explicit preferences here
        logger.info(f"NovaSpark TTV Personalization: Using explicit background preference: {theme_pref}")


    # Priority 2: Infer from User Profile (Profession/Industry/Interests)
    if profile.industry:
        ind_lower = profile.industry.lower()
        if "technology" in ind_lower or "software" in ind_lower or "data" in ind_lower:
            logger.info(f"NovaSpark TTV Personalization: Inferring tech background from industry: {profile.industry}")
            return {"type": "animated_loop_id", "id": random.choice(["abstract_tech_circuit_001", "digital_matrix_light_002"])} # Example IDs
        elif "education" in ind_lower or "academic" in ind_lower:
            logger.info(f"NovaSpark TTV Personalization: Inferring academic background from industry: {profile.industry}")
            return {"type": "preset_scene_id", "id": "calm_study_room_02"}
        elif "nature" in ind_lower or "environment" in ind_lower or ("hiking" in (profile.hobbies_and_interests or [])):
             logger.info(f"NovaSpark TTV Personalization: Inferring nature background from industry/hobbies.")
             return {"type": "image_url", "url": f"gs://{TTV_GCS_BUCKET_NAME}/backgrounds/serene_nature_landscape_01.jpg"}


    # Priority 3: Infer from Visual Cues (less direct, more abstract)
    if visual_cues:
        for cue in visual_cues:
            if "code" in cue.get("keywords","").lower() or "data" in cue.get("description","").lower():
                logger.info(f"NovaSpark TTV Personalization: Inferring tech background from visual cue: {cue}")
                return {"type": "animated_loop_id", "id": "abstract_data_stream_005"}
    
    # Fallback: Could be a default pleasant abstract or a simple color based on character.
    # For Uncle Trevor, a warm color; for Susan, a cool professional color.
    default_bg = {"type": "solid_color", "hex": "#E0E8F0"} # Neutral light blue/gray
    if request_data.instructor_character == InstructorChars.UNCLE_TREVOR:
        default_bg = {"type": "solid_color", "hex": "#F0E8E0"} # Warmer neutral
    
    logger.info(f"NovaSpark TTV Personalization: No strong preference/inference for background. Using default: {default_bg}")
    return default_bg

# --- Slideshow MVP Helper Functions (Task 1, Option B) ---
def _get_font_path(font_name="arial.ttf"):
    # For Cloud Run, package fonts with the container or use system available ones.
    # This is a placeholder for font path resolution.
    # On many Linux systems: /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
    # For local testing, ensure arial.ttf is accessible or change font.
    font_paths_to_try = [
        font_name, 
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Common on Linux
        "DejaVuSans.ttf",
        "arial.ttf" # Common fallback
    ]
    for path_attempt in font_paths_to_try:
        try:
            ImageFont.truetype(path_attempt, 10) # Test if font loads
            return path_attempt
        except IOError:
            continue
    logger.warning(f"NovaSpark Slideshow: Could not load preferred font '{font_name}'. Defaulting might occur or Pillow might error if no font found.")
    return font_name # Let Pillow try its default search path

async def _generate_text_slide_image(text_lines: List[str], output_gcs_path: str, 
                                     image_size=(1280, 720), bg_color=(20, 30, 40), text_color=(230, 230, 230)) -> Optional[str]:
    """Generates a simple text slide and uploads to GCS. Returns GCS URI or None."""
    try:
        # NovaSpark TODO: Use a more robust font loading mechanism / package fonts
        title_font_size = 60
        line_font_size = 40
        try:
            title_font = ImageFont.truetype(_get_font_path("DejaVuSans-Bold.ttf"), title_font_size)
            line_font = ImageFont.truetype(_get_font_path("DejaVuSans.ttf"), line_font_size)
        except IOError:
            logger.error("NovaSpark Slideshow: Critical font loading error. Cannot generate text slides.", exc_info=True)
            title_font = ImageFont.load_default() # Pillow's fallback
            line_font = ImageFont.load_default()


        img = Image.new('RGB', image_size, color=bg_color)
        draw = ImageDraw.Draw(img)
        
        current_y = image_size[1] * 0.2 # Start 20% from top

        for i, line_text in enumerate(text_lines):
            font_to_use = title_font if i == 0 and len(text_lines) > 1 else line_font
            # text_width, text_height = draw.textsize(line_text, font=font_to_use) # Deprecated
            bbox = draw.textbbox((0,0), line_text, font=font_to_use)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Basic text wrapping (very rudimentary)
            if text_width > image_size[0] * 0.9:
                # Simple split, could be improved with textwrap module
                words = line_text.split()
                wrapped_lines = []
                current_line = ""
                for word in words:
                    # bbox_line = draw.textbbox((0,0), current_line + word, font=font_to_use) # Inefficient
                    if len(current_line + word) * (font_to_use.size * 0.5) > image_size[0] * 0.9: # Rough width check
                        wrapped_lines.append(current_line)
                        current_line = word + " "
                    else:
                        current_line += word + " "
                wrapped_lines.append(current_line.strip())
                
                for wrapped_line_text in wrapped_lines:
                    bbox_w = draw.textbbox((0,0), wrapped_line_text, font=font_to_use)
                    text_width_w = bbox_w[2] - bbox_w[0]
                    text_height_w = bbox_w[3] - bbox_w[1]
                    draw.text(((image_size[0] - text_width_w) / 2, current_y), wrapped_line_text, font=font_to_use, fill=text_color)
                    current_y += text_height_w * 1.5 
            else:
                draw.text(((image_size[0] - text_width) / 2, current_y), line_text, font=font_to_use, fill=text_color)
                current_y += text_height * 1.8 # Increased line spacing

            if current_y > image_size[1] * 0.9: break # Stop if too much text

        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Upload to GCS
        if storage_client and TTV_GCS_BUCKET_NAME:
            bucket = storage_client.bucket(TTV_GCS_BUCKET_NAME)
            blob = bucket.blob(output_gcs_path) # e.g., "slideshow_assets/job_id/slide_01.png"
            await asyncio.to_thread(blob.upload_from_file, img_byte_arr, content_type='image/png')
            logger.info(f"NovaSpark Slideshow: Uploaded text slide to gs://{TTV_GCS_BUCKET_NAME}/{output_gcs_path}")
            return f"gs://{TTV_GCS_BUCKET_NAME}/{output_gcs_path}"
        else:
            logger.error("NovaSpark Slideshow: Storage client or bucket name not configured. Cannot upload slide.")
            return None

    except Exception as e_img:
        logger.error(f"NovaSpark Slideshow: Failed to generate or upload text slide '{output_gcs_path}'. Error: {e_img}", exc_info=True)
        return None

async def _fetch_stock_image_for_slide(keywords: str, output_gcs_path: str) -> Optional[str]:
    """Placeholder: Fetches a stock image based on keywords and uploads to GCS."""
    stock_photo_api_key = os.getenv(STOCK_PHOTO_API_KEY_ENV)
    stock_photo_base_url = os.getenv(STOCK_PHOTO_API_URL_ENV)

    if not stock_photo_api_key or not stock_photo_base_url:
        logger.warning("NovaSpark Slideshow: Stock photo API key or URL not configured. Skipping stock image fetch.")
        return None
    
    logger.info(f"NovaSpark Slideshow: Conceptually fetching stock image for keywords: '{keywords}'")
    # FIXME: Mugambi - Implement actual API call to a chosen free stock photo service (e.g., Pexels, Unsplash)
    # Ensure compliance with their API terms (attribution, rate limits).
    # Example conceptual call:
    # headers = {"Authorization": stock_photo_api_key}
    # params = {"query": keywords, "per_page": 1, "orientation": "landscape"}
    # async with httpx.AsyncClient() as client:
    #     response = await client.get(f"{stock_photo_base_url}/search", params=params, headers=headers)
    #     response.raise_for_status()
    #     image_data = response.json() # Parse response to get image URL
    #     image_url = image_data.get("photos", [{}])[0].get("src", {}).get("large")
    #     if image_url:
    #         # Download image and upload to GCS
    #         img_response = await client.get(image_url)
    #         img_bytes = BytesIO(img_response.content)
    #         # ... GCS upload logic similar to _generate_text_slide_image ...
    #         return f"gs://{TTV_GCS_BUCKET_NAME}/{output_gcs_path}" # Return GCS URI
    await asyncio.sleep(0.1) # Simulate network call
    logger.warning("NovaSpark Slideshow: Stock image fetching is a PLACEHOLDER. No actual image downloaded.")
    # For now, return a placeholder GCS path to a generic image if you have one for testing
    # return f"gs://{TTV_GCS_BUCKET_NAME}/placeholder_backgrounds/generic_placeholder.jpg" 
    return None


# --- Main Background Task ---
async def process_video_generation_task(job_id: str, request_data: GenerateVideoRequest): 
    logger.info(f"NovaSpark TTV: Starting task for job_id: {job_id}. Mode: {TTV_OPERATING_MODE}")
    video_jobs[job_id]["video_generation_mode"] = TTV_OPERATING_MODE # Store mode in job
    await update_job_status_and_notify(job_id, VideoGenerationJobStatus.PENDING) 

    raw_script_from_tutor: Optional[str] = None
    # ... (script generation and TTS steps largely same as main (14).py)
    try:
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.GENERATING_SCRIPT)
        # ... (AI Tutor call logic as in main (14).py - assumed it works and populates raw_script_from_tutor)
        script_query_for_tutor = (f"Generate a {request_data.preferred_video_length_minutes} minute video script for Uplas, explaining "
            f"{request_data.content_source.raw_text_content or ('topic ID: ' + (request_data.content_source.topic_id or 'the provided subject'))}. "
            f"The instructor persona is '{request_data.instructor_character.value}'. Personalize analogies and examples using user profile. " # Task 2 for AI Tutor is key here
            f"Include semantic tags like `<emphasis level=\"strong\">text</emphasis>`, `<pause strength=\"medium\" />`, and "
            f"`<visual_aid_suggestion type=\"[diagram|chart|image_idea]\" description=\"[description]\" keywords=\"[keywords_for_visual]\" />` where appropriate for video. " # Task 5 for AI Tutor
            f"{request_data.additional_instructions or ''}")
        tutor_user_profile_for_script = request_data.user_profile_snapshot.model_dump(exclude_none=True) # Pass full TTV profile
        # We assume AI Tutor's UserProfileSnapshot model can accept these fields or ignore extras.
        tutor_payload = {"user_id": request_data.user_id, "query_text": script_query_for_tutor,
                         "user_profile_snapshot": tutor_user_profile_for_script,
                         "language_code": request_data.language_code, "max_tokens_response": 4096, # Increased for potentially richer scripts
                         "context": {"course_id": request_data.content_source.course_id, "topic_id": request_data.content_source.topic_id}}
        async with httpx.AsyncClient(timeout=180.0, transport=httpx.AsyncHTTPTransport(retries=1)) as client:
            logger.info(f"NovaSpark TTV Job {job_id}: Calling AI Tutor at {AI_TUTOR_AGENT_URL}/v1/ask-tutor")
            response = await client.post(f"{AI_TUTOR_AGENT_URL}/v1/ask-tutor", json=tutor_payload)
            response.raise_for_status()
            tutor_response_data = response.json()
            raw_script_from_tutor = tutor_response_data.get("answer_text") # Assuming AI Tutor returns main text here
            if not raw_script_from_tutor and tutor_response_data.get("personalized_examples"): # Fallback if main answer empty but examples exist
                raw_script_from_tutor = " ".join(tutor_response_data["personalized_examples"])

        if not raw_script_from_tutor or len(raw_script_from_tutor) < 20: raise ValueError(f"AI Tutor returned insufficient script: '{raw_script_from_tutor}'")
        logger.info(f"NovaSpark TTV Job {job_id}: Script generated (len: {len(raw_script_from_tutor)}). Preview: {raw_script_from_tutor[:100]}...")
        video_jobs[job_id]["script_generated_preview"] = raw_script_from_tutor[:500]
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.PREPARING_SCRIPT)
        
        processed_ssml_for_tts = convert_script_to_ssml(raw_script_from_tutor, request_data.language_code, request_data.user_profile_snapshot.learning_pace_preference)
        visual_aid_cues = extract_visual_aid_cues(raw_script_from_tutor) 
        video_jobs[job_id]["visual_cues_identified"] = visual_aid_cues 

        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.GENERATING_AUDIO)
        # ... (TTS Agent call logic as in main (14).py - assumed it works and populates generated_audio_gcs_url, tts_duration_seconds)
        tts_voice_char_map = {InstructorChars.SUSAN: "susan_us_standard", InstructorChars.UNCLE_TREVOR: "trevor_us_standard"} # Simplified for example
        # Task 2: Potentially refine tts_voice_character based on user_profile.preferred_visual_style or other persona cues if desired for TTS also
        tts_voice_character_name_for_tts = request_data.instructor_character.value + f"_{request_data.language_code.split('-')[0].lower()}" # e.g. susan_us, trevor_fr
        # A more robust mapping from Uplas InstructorChars to TTS voice_character_names (in tts_agent) would be needed.
        # For now, using a simplified version.
        if request_data.instructor_character == InstructorChars.SUSAN : tts_voice_character_name_for_tts = f"susan_{request_data.language_code.split('-')[0].lower()}" # e.g. susan_us
        elif request_data.instructor_character == InstructorChars.UNCLE_TREVOR: tts_voice_character_name_for_tts = f"trevor_{request_data.language_code.split('-')[0].lower()}"
        else: tts_voice_character_name_for_tts = f"default_{request_data.language_code.split('-')[0].lower()}"


        tts_payload = {"input_type": TtsSynthesisInputType.SSML.value, "content_to_synthesize": processed_ssml_for_tts,
                       "language_code": request_data.language_code, # TTS agent will use this to pick appropriate voice if char name isn't lang specific
                       "voice_params": {"voice_character_name": tts_voice_character_name_for_tts }, # TTS agent maps this
                       "audio_config": {"audio_encoding": "MP3"}} # Transcoder API generally prefers MP3 or common formats
        async with httpx.AsyncClient(timeout=90.0, transport=httpx.AsyncHTTPTransport(retries=1)) as client:
            logger.info(f"NovaSpark TTV Job {job_id}: Calling TTS Agent at {TTS_AGENT_URL}/v1/synthesize-speech")
            response = await client.post(f"{TTS_AGENT_URL}/v1/synthesize-speech", json=tts_payload)
            response.raise_for_status()
            tts_response_data = response.json()
            generated_audio_gcs_url = tts_response_data.get("audio_url")
            tts_duration_seconds = tts_response_data.get("audio_duration_seconds")
        if not generated_audio_gcs_url: raise ValueError("TTS Agent did not return an audio URL.")
        logger.info(f"NovaSpark TTV Job {job_id}: Audio generated: {generated_audio_gcs_url}, Duration: {tts_duration_seconds}s")
        
        # Store character name for callback, regardless of mode
        video_jobs[job_id]["character_used"] = request_data.instructor_character.value
        
        # --- NovaSpark: Conditional Workflow based on TTV_OPERATING_MODE ---
        if TTV_OPERATING_MODE == "FULL_AVATAR":
            if not avatar_service_client: 
                raise EnvironmentError("Avatar Service Client not available for FULL_AVATAR mode. Check TTV_OPERATING_MODE and API key configurations.")
            await update_job_status_and_notify(job_id, VideoGenerationJobStatus.SUBMITTING_TO_AVATAR_SERVICE, audio_gcs_url_temp=generated_audio_gcs_url)
            # ... (Existing logic for FULL_AVATAR mode from main (14).py - submit_video_creation_job, poll_video_job_status)
            # This part uses determine_attire_tags and determine_background_settings which are now enhanced for personalization (Task 2)
            try:
                character_service_avatar_id = get_character_avatar_id_from_service(request_data.instructor_character.value)
                # Task 2: Pass request_data to determine_attire_tags for personalization
                attire_tags = determine_attire_tags(request_data) 
                chosen_attire_id = get_character_attire_id(instructor_character_name=request_data.instructor_character.value, preferred_attire_name=request_data.preferred_attire_name, tags=attire_tags)
                video_jobs[job_id]["attire_used"] = chosen_attire_id or "CHAR_DEFAULT_ATTIRE"
                # Task 2: Pass request_data to determine_background_settings for personalization
                bg_settings = determine_background_settings(request_data, visual_aid_cues) 
                if bg_settings: video_jobs[job_id]["background_details"] = bg_settings

            except CharacterConfigError as e: logger.error(f"Job {job_id}: Character config error: {e}", exc_info=True); raise ValueError(f"Invalid char/attire config: {e}")
            
            logger.info(f"Job {job_id}: Submitting to Avatar Service. AvatarID: {character_service_avatar_id}, AttireID: {chosen_attire_id}, Audio: {generated_audio_gcs_url}")
            service_job_details = await avatar_service_client.submit_video_creation_job(
                service_avatar_id=character_service_avatar_id, audio_file_gcs_url=generated_audio_gcs_url,
                script_text_for_lipsync_data=raw_script_from_tutor, # Pass raw script if service can use it for better lipsync data
                service_attire_id=chosen_attire_id, language_code=request_data.language_code,
                background_settings=bg_settings, custom_metadata={"uplas_job_id": job_id, "user_id": request_data.user_id}
            )
            third_party_job_id = service_job_details.get("service_job_id")
            if not third_party_job_id: raise AvatarJobError("Avatar service did not return a job ID.")
            video_jobs[job_id]["third_party_avatar_job_id"] = third_party_job_id
            await update_job_status_and_notify(job_id, VideoGenerationJobStatus.RENDERING_AVATAR_VIDEO, third_party_job_id=third_party_job_id)
            
            # Polling (simplified from main (14).py for brevity, assume it works)
            final_video_url, final_thumbnail_url, video_duration_final = None, None, tts_duration_seconds
            max_polls = int(os.getenv("AVATAR_MAX_POLL_ATTEMPTS", 60)); poll_interval = int(os.getenv("AVATAR_POLL_INTERVAL_SECONDS", 30))
            await asyncio.sleep(int(os.getenv("AVATAR_INITIAL_POLL_DELAY", 10)))
            for attempt in range(max_polls):
                logger.info(f"Job {job_id}: Polling avatar service for job {third_party_job_id}, attempt {attempt + 1}/{max_polls}")
                status_resp = await avatar_service_client.poll_video_job_status(third_party_job_id)
                service_status = status_resp.get("status", "unknown").lower()
                if service_status == "completed":
                    final_video_url = status_resp.get("videoUrl"); final_thumbnail_url = status_resp.get("thumbnailUrl")
                    video_duration_final = status_resp.get("durationSeconds", video_duration_final)
                    if not final_video_url: raise AvatarJobError("Avatar service 'completed' but no video URL."); break
                elif service_status == "failed": raise AvatarJobError(f"Avatar generation failed: {status_resp.get('errorMessage', 'Unknown service error')}")
                elif service_status in ["processing", "queued", "rendering"]: await asyncio.sleep(poll_interval)
                else: logger.warning(f"Job {job_id}: Unknown status '{service_status}'. Retrying."); await asyncio.sleep(poll_interval * 2)
            if not final_video_url: raise TimeoutError("Timeout waiting for avatar video.")
            
            await update_job_status_and_notify(job_id, VideoGenerationJobStatus.COMPLETED, video_url=final_video_url, thumbnail_url=final_thumbnail_url, video_duration_seconds=round(video_duration_final,2) if video_duration_final else None)

        elif TTV_OPERATING_MODE == "SLIDESHOW_MVP":
            if not transcoder_client or not storage_client or not TTV_GCS_BUCKET_NAME:
                raise EnvironmentError("GCP Transcoder/Storage client or GCS bucket not configured for SLIDESHOW_MVP mode.")
            await update_job_status_and_notify(job_id, VideoGenerationJobStatus.GENERATING_SLIDES)
            
            slide_gcs_uris = []
            slide_default_duration_seconds = 5 # Default duration per slide if not tied to audio segments

            # 1. Title Slide
            title_slide_text = [request_data.content_source.raw_text_content[:80] or "Uplas Learning Video", f"By {request_data.instructor_character.value.replace('_', ' ').title()}"]
            title_slide_gcs_path = f"slideshow_assets/{job_id}/slide_title.png"
            title_slide_uri = await _generate_text_slide_image(title_slide_text, title_slide_gcs_path)
            if title_slide_uri: slide_gcs_uris.append({"uri": title_slide_uri, "duration": slide_default_duration_seconds})

            # 2. Slides from Visual Cues / Script Segments
            # This is a simplified approach. A more robust solution would segment the script
            # and create slides with text corresponding to audio segments.
            max_slides_from_cues = 3 # Limit number of cue-based slides for MVP
            for i, cue in enumerate(visual_aid_cues[:max_slides_from_cues]):
                cue_slide_text = [cue.get("description", "Visual Aid"), cue.get("type", "")]
                cue_slide_gcs_path = f"slideshow_assets/{job_id}/slide_cue_{i+1}.png"
                
                # Try to fetch a stock image for this cue conceptually
                stock_image_gcs_uri = None
                if cue.get("keywords"):
                    stock_image_gcs_path = f"slideshow_assets/{job_id}/slide_cue_{i+1}_stock.jpg"
                    stock_image_gcs_uri = await _fetch_stock_image_for_slide(cue["keywords"], stock_image_gcs_path)

                if stock_image_gcs_uri: # If stock image found, use it
                     slide_gcs_uris.append({"uri": stock_image_gcs_uri, "duration": slide_default_duration_seconds + 2})
                elif cue_slide_text[0] != "Visual Aid": # Else generate text slide if description is meaningful
                    text_slide_uri = await _generate_text_slide_image(cue_slide_text, cue_slide_gcs_path)
                    if text_slide_uri: slide_gcs_uris.append({"uri": text_slide_uri, "duration": slide_default_duration_seconds})
            
            if not slide_gcs_uris: # Ensure at least one slide if cues were empty
                fallback_slide_text = ["Key Concepts", raw_script_from_tutor[:100] + "..."]
                fallback_slide_gcs_path = f"slideshow_assets/{job_id}/slide_fallback.png"
                fallback_slide_uri = await _generate_text_slide_image(fallback_slide_text, fallback_slide_gcs_path)
                if fallback_slide_uri: slide_gcs_uris.append({"uri": fallback_slide_uri, "duration": slide_default_duration_seconds})

            if not slide_gcs_uris or not generated_audio_gcs_url:
                raise ValueError("Failed to generate slides or audio for SLIDESHOW_MVP.")

            # 3. Submit to GCP Transcoder API for stitching
            await update_job_status_and_notify(job_id, VideoGenerationJobStatus.SUBMITTING_TO_TRANSCODER)
            output_video_gcs_path = f"final_videos/{job_id}/stitched_video.mp4"
            
            # Construct Transcoder Job (Simplified example)
            # Actual implementation needs careful mapping of slide durations to audio length.
            # For MVP, we can just sequence images with fixed duration each.
            edit_list = []
            total_slide_duration = 0
            for slide_item in slide_gcs_uris:
                edit_list.append({
                    "key": f"slide-{uuid.uuid4().hex[:4]}", # Unique key for edit atom
                    "inputs": [{"key": f"image-input-{uuid.uuid4().hex[:4]}", "uri": slide_item["uri"]}],
                    "startTimeOffset": {"seconds": total_slide_duration},
                    "endTimeOffset": {"seconds": total_slide_duration + slide_item["duration"]}
                })
                total_slide_duration += slide_item["duration"]
            
            # Ensure audio duration is at least total_slide_duration, or cap slides
            # For a robust solution, segment audio and align precisely.
            # Here, we just overlay the full audio track.
            job_config = {
                "inputs": [{"key": "audioTrack", "uri": generated_audio_gcs_url}], # Audio input
                "edit_list": edit_list,
                "elementary_streams": [ # Define how to process images into a video stream
                    {
                        "key": "video-stream",
                        "video_stream": {
                            "h264": { "height_pixels": 720, "width_pixels": 1280, "bitrate_bps": 2500000, "frame_rate": 25 }
                        }
                    }
                ],
                "mux_streams": [ # Combine video and audio into an MP4
                    {
                        "key": "mp4-output", "file_name": output_video_gcs_path, "container": "mp4",
                        "elementary_streams": ["video-stream", "audioTrack"] # Use the key for audio input if it's just one track
                    }
                ],
                "output": {"uri": f"gs://{TTV_GCS_BUCKET_NAME}/"}, # Output directory
                 "config_id": f"uplas-slideshow-mvp-preset-{uuid.uuid4().hex[:4]}" # Optional descriptive config ID
            }
            # If just one audio input, we reference its key in elementary_streams if it's part of edit_list,
            # or directly in mux_streams if it's an overlay for the whole edit_list.
            # The Transcoder API is flexible but requires careful config.
            # Simpler: If audio is just one track for the whole video:
            job_config_simple_audio_overlay = {
                "edit_list": edit_list, # List of image "atoms"
                "elementary_streams": [ # Define video stream from images
                     { "key": "video-stream", "video_stream": { "h264": { "height_pixels": 720, "width_pixels": 1280, "bitrate_bps": 2500000, "frame_rate": 25 }}}
                ],
                "audio_streams": [ # Define how to process audio input
                    {"key": "audio-overlay", "mapping": [{"atom_key": "audioTrack", "input_key": "audioTrack", "input_track": 0 }]} # Map input to a stream key
                ],
                "inputs": [{"key": "audioTrack", "uri": generated_audio_gcs_url}], # Audio input
                "mux_streams": [
                    { "key": "mp4-output", "file_name": output_video_gcs_path, "container": "mp4",
                      "elementary_streams": ["video-stream", "audio-overlay"]} # Combine video and audio streams
                ],
                "output": {"uri": f"gs://{TTV_GCS_BUCKET_NAME}/"},
                "config_id": f"uplas-slideshow-mvp-preset-{uuid.uuid4().hex[:4]}"
            }
            # This config is still complex. Actual Transcoder Job:
            parent = f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}"
            transcoder_job = transcoder_v1.types.Job()
            transcoder_job.output_uri = f"gs://{TTV_GCS_BUCKET_NAME}/{output_video_gcs_path.rsplit('/', 1)[0]}/" # Directory for output

            # Constructing config inputs based on each image
            config_inputs = []
            for i, slide in enumerate(slide_gcs_uris):
                config_inputs.append(transcoder_v1.types.Input(key=f"image-{i}", uri=slide["uri"]))
            config_inputs.append(transcoder_v1.types.Input(key="main_audio", uri=generated_audio_gcs_url))
            transcoder_job.inputs.extend(config_inputs)

            # Construct Edit List
            atom_key_counter = 0
            edit_list_atoms = []
            current_time_offset_seconds = 0
            for slide in slide_gcs_uris:
                atom_key = f"atom-{atom_key_counter}"
                atom_key_counter += 1
                edit_list_atoms.append(transcoder_v1.types.EditAtom(
                    key=atom_key,
                    inputs=[f"image-{slide_gcs_uris.index(slide)}"], # Reference key from transcoder_job.inputs
                    start_time_offset={"seconds": current_time_offset_seconds},
                    end_time_offset={"seconds": current_time_offset_seconds + slide["duration"]}
                ))
                current_time_offset_seconds += slide["duration"]
            
            transcoder_job.edit_list.extend(edit_list_atoms)
            
            # Define elementary streams (video from images, audio from audio input)
            transcoder_job.elementary_streams.append(transcoder_v1.types.ElementaryStream(
                key="video_stream0",
                video_stream=transcoder_v1.types.VideoStream(
                    codec="h264", height_pixels=720, width_pixels=1280, frame_rate=25, bitrate_bps=2500000
                )
            ))
            transcoder_job.elementary_streams.append(transcoder_v1.types.ElementaryStream(
                key="audio_stream0",
                audio_stream=transcoder_v1.types.AudioStream(codec="mp3", bitrate_bps=128000) # Output audio stream config
            ))
             # Define MuxStreams to combine into final output
            transcoder_job.mux_streams.append(transcoder_v1.types.MuxStream(
                key="video_output", file_name=output_video_gcs_path.split('/')[-1], container="mp4", # just filename
                elementary_streams=["video_stream0", "audio_stream0"]
            ))
            
            # Create the job
            logger.info(f"NovaSpark TTV Job {job_id}: Submitting Transcoder job. Config (partial): {transcoder_job.output_uri}")
            transcoder_response = await asyncio.to_thread(transcoder_client.create_job, parent=parent, job=transcoder_job)
            transcoder_job_name = transcoder_response.name # Full job name path
            video_jobs[job_id]["gcp_transcoder_job_name"] = transcoder_job_name
            await update_job_status_and_notify(job_id, VideoGenerationJobStatus.RENDERING_TRANSCODED_VIDEO, gcp_transcoder_job_name=transcoder_job_name)

            # 4. Poll GCP Transcoder Job
            final_video_url_transcoded = None
            max_transcoder_polls = int(os.getenv("TRANSCODER_MAX_POLL_ATTEMPTS", 40)) # e.g. 10 mins if 15s interval
            transcoder_poll_interval = int(os.getenv("TRANSCODER_POLL_INTERVAL_SECONDS", 15))

            for attempt in range(max_transcoder_polls):
                logger.info(f"Job {job_id}: Polling Transcoder job {transcoder_job_name}, attempt {attempt+1}/{max_transcoder_polls}")
                job_details = await asyncio.to_thread(transcoder_client.get_job, name=transcoder_job_name)
                if job_details.state == transcoder_v1.types.Job.ProcessingState.SUCCEEDED:
                    final_video_url_transcoded = f"gs://{TTV_GCS_BUCKET_NAME}/{output_video_gcs_path}" # Construct final GCS URL
                    # Transcoder doesn't directly give duration of stitched output easily in job_details, use TTS estimate
                    logger.info(f"Job {job_id}: Transcoded video completed! URL: {final_video_url_transcoded}")
                    break
                elif job_details.state == transcoder_v1.types.Job.ProcessingState.FAILED:
                    error_msg = f"Transcoder job failed: {job_details.error.message if job_details.error else 'Unknown transcoder error'}"
                    logger.error(f"Job {job_id}: {error_msg}")
                    raise Exception(error_msg)
                elif job_details.state == transcoder_v1.types.Job.ProcessingState.PENDING or job_details.state == transcoder_v1.types.Job.ProcessingState.RUNNING:
                    logger.info(f"Job {job_id}: Transcoder status '{job_details.state.name}'. Polling again in {transcoder_poll_interval}s...")
                    await asyncio.sleep(transcoder_poll_interval)
                else: # Should not happen
                    logger.warning(f"Job {job_id}: Unknown Transcoder state '{job_details.state.name}'. Retrying.")
                    await asyncio.sleep(transcoder_poll_interval * 2)
            
            if not final_video_url_transcoded:
                raise TimeoutError(f"Timeout waiting for Transcoder job {transcoder_job_name}.")

            await update_job_status_and_notify(job_id, VideoGenerationJobStatus.COMPLETED, video_url=final_video_url_transcoded, video_duration_seconds=tts_duration_seconds)

        else: # Unknown TTV_OPERATING_MODE
            raise ValueError(f"Unsupported TTV_OPERATING_MODE: {TTV_OPERATING_MODE}")
            
    except ValueError as ve: 
        logger.error(f"TTV Job {job_id}: Validation error: {ve}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=f"Input validation error: {str(ve)}")
    except httpx.HTTPStatusError as hse: 
        logger.error(f"TTV Job {job_id}: HTTP error calling service {hse.request.url}: {hse.response.status_code}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=f"Dependency service error ({hse.request.url}): {hse.response.status_code}")
    except AvatarJobError as ae: 
        logger.error(f"TTV Job {job_id}: Avatar service error: {ae}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=f"Avatar service error: {str(ae)}")
    except TimeoutError as te: 
        logger.error(f"TTV Job {job_id}: Timeout error: {te}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=str(te))
    except Exception as e: 
        logger.error(f"TTV Job {job_id}: Unhandled exception: {e}", exc_info=True)
        await update_job_status_and_notify(job_id, VideoGenerationJobStatus.FAILED, error_message=f"An unexpected error occurred: {str(e)}")


# --- API Endpoints ---
@app.post("/v1/generate-video", response_model=GenerateVideoInitialResponse, status_code=status.HTTP_202_ACCEPTED) 
async def generate_video_endpoint(request_data: GenerateVideoRequest, background_tasks: BackgroundTasks):
    # Basic config check (can be more elaborate in health check)
    if TTV_OPERATING_MODE == "FULL_AVATAR" and (not THIRD_PARTY_AVATAR_API_KEY or not THIRD_PARTY_AVATAR_BASE_URL) and not (os.getenv("ALLOW_MOCK_MODE_IF_UNCONFIGURED", "false").lower() == "true" and avatar_service_client and avatar_service_client.is_mock_mode):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="TTV service (FULL_AVATAR mode) is not properly configured for real calls.")
    if TTV_OPERATING_MODE == "SLIDESHOW_MVP" and (not GCP_PROJECT_ID or not TTV_GCS_BUCKET_NAME or not transcoder_client):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="TTV service (SLIDESHOW_MVP mode) is not properly configured with GCP resources.")


    job_id = f"ttvjob_{uuid.uuid4()}"
    video_jobs[job_id] = { 
        "status": VideoGenerationJobStatus.PENDING, "requested_at": time.time(),
        "user_id": request_data.user_id, "instructor_character_requested": request_data.instructor_character.value,
        "language_requested": request_data.language_code, "video_generation_mode_requested": TTV_OPERATING_MODE,
        # ... other initial fields
    }
    background_tasks.add_task(process_video_generation_task, job_id, request_data) 
    est_time = 10 if TTV_OPERATING_MODE == "FULL_AVATAR" else 3 # Different estimates
    return GenerateVideoInitialResponse(job_id=job_id, status=VideoGenerationJobStatus.PENDING, estimated_completion_time_minutes=est_time, message=f"Video generation task accepted ({TTV_OPERATING_MODE} mode).")


@app.get("/v1/video-status/{job_id}", response_model=VideoCallbackPayload, summary="Get status of a video generation job") 
async def get_video_status_endpoint(job_id: str):
    # ... (endpoint from main (14).py - add video_generation_mode to response)
    job_info = video_jobs.get(job_id)
    if not job_info: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")
    payload_dict = {
        "job_id":job_id, "status":job_info.get("status", VideoGenerationJobStatus.FAILED),
        "video_url":job_info.get("video_url"), "thumbnail_url":job_info.get("thumbnail_url"),
        "error_message":job_info.get("error_message"), "video_duration_seconds":job_info.get("video_duration_seconds"),
        "script_generated_preview":job_info.get("script_generated_preview"), "character_used":job_info.get("character_used"),
        "attire_used":job_info.get("attire_used"), "video_generation_mode": job_info.get("video_generation_mode", TTV_OPERATING_MODE)
    }
    return VideoCallbackPayload(**payload_dict)

@app.get("/health", status_code=status.HTTP_200_OK) 
async def health_check():
    # ... (health check from main (14).py, slightly adapted)
    missing_configs = []
    service_name = f"TTV_Agent_NovaSpark ({TTV_OPERATING_MODE})"
    if not AI_TUTOR_AGENT_URL: missing_configs.append("AI_TUTOR_AGENT_URL")
    if not TTS_AGENT_URL: missing_configs.append("TTS_AGENT_URL")
    if not DJANGO_TTV_CALLBACK_URL: missing_configs.append("DJANGO_TTV_CALLBACK_URL")
    if not GCP_PROJECT_ID: missing_configs.append("GCP_PROJECT_ID")
    if not TTV_GCS_BUCKET_NAME: missing_configs.append("TTV_GCS_BUCKET_NAME")

    if TTV_OPERATING_MODE == "FULL_AVATAR":
        if avatar_service_client and not avatar_service_client.is_mock_mode and (not THIRD_PARTY_AVATAR_API_KEY or not THIRD_PARTY_AVATAR_BASE_URL):
            missing_configs.append("THIRD_PARTY_AVATAR_API_KEY/BASE_URL (for non-mock FULL_AVATAR mode)")
    elif TTV_OPERATING_MODE == "SLIDESHOW_MVP":
        if not transcoder_client: missing_configs.append("GCP_TRANSCODER_CLIENT (not initialized for SLIDESHOW_MVP)")
    else:
        missing_configs.append(f"Unsupported TTV_OPERATING_MODE: {TTV_OPERATING_MODE}")


    if missing_configs:
        return {"status": "unhealthy", "reason": f"Missing configurations: {', '.join(missing_configs)}", "service": service_name, "innovate_ai_enhancements_active": True}
    return {"status": "healthy", "service": service_name, "innovate_ai_enhancements_active": True}


if __name__ == "__main__": 
    import uvicorn
    logger.info(f"NovaSpark TTV: Starting Agent in {TTV_OPERATING_MODE} mode for local development...")
    # ... (startup print warnings from main (14).py)
    if not all([GCP_PROJECT_ID, TTV_GCS_BUCKET_NAME, AI_TUTOR_AGENT_URL, TTS_AGENT_URL, DJANGO_TTV_CALLBACK_URL]):
        print("NovaSpark TTV WARNING: One or more critical base environment variables are not set.")
    if TTV_OPERATING_MODE == "FULL_AVATAR" and avatar_service_client and not avatar_service_client.is_mock_mode and (not THIRD_PARTY_AVATAR_API_KEY or not THIRD_PARTY_AVATAR_BASE_URL):
        print("NovaSpark TTV WARNING: FULL_AVATAR mode is active but Third Party Avatar API Key/URL might be missing for real calls.")
    if TTV_OPERATING_MODE == "SLIDESHOW_MVP" and not transcoder_client :
         print("NovaSpark TTV WARNING: SLIDESHOW_MVP mode is active but GCP Transcoder Client failed to initialize.")

    port = int(os.getenv("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
