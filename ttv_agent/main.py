from fastapi import FastAPI, HTTPException, BackgroundTasks, Request as FastAPIRequest
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import os
import uuid
import time # For processing time and job IDs
import random # For mock elements

# --- Pydantic Models for API Contract ---

class UserProfileSnapshotForTTV(BaseModel): # Subset of UserProfile relevant to TTV
    industry: Optional[str] = Field(None, examples=["Healthcare"])
    profession: Optional[str] = Field(None, examples=["Nurse Practitioner"])
    country: Optional[str] = Field(None, examples=["Canada"])
    city: Optional[str] = Field(None, examples=["Toronto"])
    # Add learning_style_preference if it influences visual presentation
    # learning_style_preference: Optional[Dict[str, float]] = Field(default_factory=dict)
    preferred_ttv_instructor: Optional[str] = Field(None, examples=["susan"]) # uncle_trevor or susan
    # preferred_ttv_instructor_attire: Optional[str] = Field(None, examples=["professional_everyday", "lab_coat_uncle_trevor"])


class ContentSource(BaseModel):
    topic_id: Optional[str] = Field(None, examples=["topic_uuid_for_python_loops"])
    raw_text_content: Optional[str] = Field(None, examples=["Explain Python loops with an example for a beginner in finance."])
    # Potentially add course_id if topic_id alone isn't enough context for Django content fetching
    course_id: Optional[str] = None

    @validator('raw_text_content', always=True)
    def check_content_provided(cls, v, values):
        if not v and not values.get('topic_id'):
            raise ValueError('Either topic_id or raw_text_content must be provided.')
        return v


class InstructorCharacterEnum(str, Enum):
    UNCLE_TREVOR = "uncle_trevor"
    SUSAN = "susan"


class GenerateVideoRequest(BaseModel):
    user_id: str = Field(..., examples=["user_uuid_for_video_gen"])
    content_source: ContentSource
    instructor_character: InstructorCharacterEnum = Field(InstructorCharacterEnum.SUSAN)
    # User preference can be overridden by this request
    # user_profile_snapshot: UserProfileSnapshotForTTV # Full profile from Django call
    # For TTV, we might only need a few key fields for script personalization directly in this request
    # The Django backend would construct this based on the full user profile.
    personalization_hints: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"industry": "Finance", "profession": "Analyst"}])
    preferred_video_length_minutes: Optional[str] = Field("3-5", examples=["2-3", "4-6"])
    # Add language if videos can be in different languages
    language_code: Optional[str] = Field("en-US", examples=["en-US", "es-ES"])


class VideoGenerationJobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING_SCRIPT = "processing_script"
    PROCESSING_AUDIO = "processing_audio"
    PROCESSING_AVATAR = "processing_avatar_video"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateVideoInitialResponse(BaseModel):
    job_id: str = Field(..., examples=[f"ttvjob_{uuid.uuid4()}"])
    status: VideoGenerationJobStatus = Field(VideoGenerationJobStatus.PENDING)
    message: str = Field("Video generation task accepted and queued.")
    estimated_completion_time_minutes: Optional[int] = Field(None, examples=[5, 10]) # Rough estimate


class VideoCallbackPayload(BaseModel): # For when video is ready (TTV agent calls Django)
    job_id: str
    status: VideoGenerationJobStatus
    video_url: Optional[HttpUrl] = None # Using Pydantic's HttpUrl for validation
    thumbnail_url: Optional[HttpUrl] = None
    error_message: Optional[str] = None
    video_duration_seconds: Optional[float] = None


# --- FastAPI Application ---
app = FastAPI(
    title="Uplas Text-to-Video (TTV) Agent",
    description="Orchestrates script generation, TTS, and avatar video synthesis (all mocked).",
    version="0.1.0"
)

# --- Mock External Service Clients ---
MOCKED_GCS_VIDEO_BUCKET = "mocked-uplas-ttv-videos"
MOCKED_DJANGO_CALLBACK_URL = os.getenv("DJANGO_TTV_CALLBACK_URL", "http://localhost:8000/api/internal/ttv-callback/") # Placeholder

# In-memory "job store" for this mock agent
video_jobs: Dict[str, Dict[str, Any]] = {}

class MockLLMForScriptClient:
    async def generate_script(self, text_topic: str, personalization_hints: Dict, instructor: str, length_minutes: str, lang: str) -> str:
        # Simulate script generation
        print(f"MockLLM(Script): Generating script for '{text_topic[:30]}...' for {instructor} ({length_minutes} min, lang={lang}) with hints: {personalization_hints}")
        persona_script_intro = ""
        if instructor == InstructorCharacterEnum.UNCLE_TREVOR:
            persona_script_intro = "Alright folks, Uncle Trevor here! Let's dive into this. "
            if "Finance" in personalization_hints.get("industry", ""):
                persona_script_intro += "Now, if you're in finance, this is like managing your portfolio... "
        elif instructor == InstructorCharacterEnum.SUSAN:
            persona_script_intro = "Hello everyone, I'm Susan. Today, we'll explore an exciting concept. "
            if "Healthcare" in personalization_hints.get("industry", ""):
                 persona_script_intro += "For those in healthcare, this is similar to diagnosing a patient based on symptoms... "

        script = f"""
        [SCENE START]
        VISUAL: Title Card: Explanation of '{text_topic[:20]}'
        {instructor.upper()}: {persona_script_intro}Today we are talking about '{text_topic}'.

        [SCENE BREAK]
        VISUAL: {instructor} on screen, dynamic background related to {personalization_hints.get("industry", "a general topic")}.
        {instructor.upper()}: This concept is particularly useful for a {personalization_hints.get("profession", "learner")}. For instance... [explains concept with a simple example].

        [SCENE BREAK]
        VISUAL: On-screen text highlighting key points.
        {instructor.upper()}: Remember these key takeaways: 1. Point one. 2. Point two.

        [SCENE END]
        {instructor.upper()}: I hope that clarifies things! Keep learning!
        """
        # await asyncio.sleep(random.uniform(0.5, 1.5)) # Simulate LLM processing
        return script

class MockTTSForVideoClient: # Could eventually call our TTS Agent 2
    async def generate_audio_from_script(self, script: str, instructor: str, lang: str) -> Dict[str, Any]:
        # Simulate TTS, return path to a (mock) audio file and duration
        print(f"MockTTS(Video): Generating audio for {instructor} (lang={lang}) from script: '{script[:50]}...'")
        audio_filename = f"audio_{instructor}_{uuid.uuid4()}.mp3"
        mock_audio_path_gcs = f"gs://{MOCKED_GCS_VIDEO_BUCKET}/intermediate_audio/{audio_filename}"
        
        # Simulate audio content being generated and stored
        # In a real scenario, actual audio bytes would be created.
        # For mock, we just generate a path.
        
        # Estimate duration based on script length (very rough)
        duration_seconds = len(script.split()) / 2.5 # Approx 2.5 words per second
        # await asyncio.sleep(random.uniform(1.0, 3.0)) # Simulate TTS processing
        return {"gcs_audio_url": mock_audio_path_gcs, "duration_seconds": round(duration_seconds,2)}

class MockAvatarVideoClient: # Simulates D-ID, Synthesia, etc.
    async def generate_video_from_audio_and_avatar(
        self, audio_url: str, instructor: str, instructor_attire: Optional[str] = None
    ) -> Dict[str, Any]:
        print(f"MockAvatarVideo: Generating video for {instructor} (attire: {instructor_attire or 'default'}) using audio: {audio_url}")
        video_filename = f"video_{instructor}_{uuid.uuid4()}.mp4"
        mock_video_path_gcs = f"https://storage.googleapis.com/{MOCKED_GCS_VIDEO_BUCKET}/final_videos/{video_filename}"
        mock_thumbnail_path_gcs = f"https://storage.googleapis.com/{MOCKED_GCS_VIDEO_BUCKET}/final_videos/thumbnails/thumb_{video_filename}.jpg"
        
        # Simulate video generation time based on audio length (if we had it)
        # await asyncio.sleep(random.uniform(5.0, 15.0)) # Simulate video rendering
        return {"gcs_video_url": mock_video_path_gcs, "gcs_thumbnail_url": mock_thumbnail_path_gcs}

# Initialize mock clients
mock_script_llm = MockLLMForScriptClient()
mock_tts_video = MockTTSForVideoClient()
mock_avatar_client = MockAvatarVideoClient()

# --- Background Task for Video Generation ---
async def process_video_generation_task(job_id: str, request_data: GenerateVideoRequest):
    """
    The actual video generation pipeline (all steps mocked).
    This runs in the background.
    """
    global video_jobs
    video_jobs[job_id]["status"] = VideoGenerationJobStatus.PROCESSING_SCRIPT
    print(f"TTV Job {job_id}: Started processing.")

    try:
        # 1. Fetch content if topic_id provided (mocked call to Django)
        actual_content_to_explain = request_data.content_source.raw_text_content
        if not actual_content_to_explain and request_data.content_source.topic_id:
            # actual_content_to_explain = await fetch_topic_content_from_django(request_data.content_source.topic_id)
            actual_content_to_explain = f"Detailed content for topic {request_data.content_source.topic_id} (fetched from mock Django)."
            if not actual_content_to_explain:
                raise ValueError(f"Could not fetch content for topic_id: {request_data.content_source.topic_id}")
        
        # 2. Generate script
        script = await mock_script_llm.generate_script(
            text_topic=actual_content_to_explain,
            personalization_hints=request_data.personalization_hints,
            instructor=request_data.instructor_character.value,
            length_minutes=request_data.preferred_video_length_minutes,
            lang=request_data.language_code
        )
        video_jobs[job_id]["status"] = VideoGenerationJobStatus.PROCESSING_AUDIO
        video_jobs[job_id]["generated_script_preview"] = script[:200] # Store preview

        # 3. Generate TTS from script
        audio_info = await mock_tts_video.generate_audio_from_script(
            script=script,
            instructor=request_data.instructor_character.value,
            lang=request_data.language_code
        )
        video_jobs[job_id]["status"] = VideoGenerationJobStatus.PROCESSING_AVATAR
        video_jobs[job_id]["audio_url_temp"] = audio_info["gcs_audio_url"]
        video_duration_seconds = audio_info["duration_seconds"]


        # 4. Generate Avatar Video from TTS audio
        # Simulate different attire based on day or preference (if passed)
        attire_options = {"uncle_trevor": ["casual_ut", "professional_ut"], "susan": ["smart_casual_s", "formal_s"]}
        chosen_attire = random.choice(attire_options.get(request_data.instructor_character.value, ["default"]))
        
        video_output = await mock_avatar_client.generate_video_from_audio_and_avatar(
            audio_url=audio_info["gcs_audio_url"],
            instructor=request_data.instructor_character.value,
            instructor_attire=chosen_attire
        )
        
        final_video_url = video_output["gcs_video_url"]
        final_thumbnail_url = video_output["gcs_thumbnail_url"]
        video_jobs[job_id].update({
            "status": VideoGenerationJobStatus.COMPLETED,
            "video_url": final_video_url,
            "thumbnail_url": final_thumbnail_url,
            "video_duration_seconds": video_duration_seconds,
            "error_message": None
        })
        print(f"TTV Job {job_id}: Successfully completed. Video URL: {final_video_url}")

    except Exception as e:
        print(f"TTV Job {job_id}: Failed. Error: {e}")
        video_jobs[job_id].update({
            "status": VideoGenerationJobStatus.FAILED,
            "error_message": str(e)
        })
        final_video_url = None # Ensure it's None on failure
        final_thumbnail_url = None
        video_duration_seconds = None


    # 5. Send callback to Django (mocked)
    # In a real system, use a robust task queue (Celery, RabbitMQ, Pub/Sub) for this callback.
    callback_payload = VideoCallbackPayload(
        job_id=job_id,
        status=video_jobs[job_id]["status"],
        video_url=final_video_url, # Pydantic will validate if it's a valid HttpUrl or None
        thumbnail_url=final_thumbnail_url,
        error_message=video_jobs[job_id].get("error_message"),
        video_duration_seconds=video_duration_seconds
    )
    try:
        # This is a fire-and-forget call for the mock. Production needs retries/dead-letter queue.
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(MOCKED_DJANGO_CALLBACK_URL, json=callback_payload.model_dump())
        #     response.raise_for_status() # Check for callback errors
        print(f"TTV Job {job_id}: Simulated callback sent to Django with payload: {callback_payload.model_dump_json(indent=2)}")
    except Exception as cb_e:
        print(f"TTV Job {job_id}: Failed to send callback to Django. Error: {cb_e}")
        # Log this failure, might need manual reconciliation.


# --- API Endpoints ---

@app.post("/v1/generate-video", response_model=GenerateVideoInitialResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_video_endpoint(request_data: GenerateVideoRequest, background_tasks: BackgroundTasks):
    """
    Accepts a request to generate an instructor video.
    Starts the generation process in the background and returns a job ID.
    """
    job_id = f"ttvjob_{uuid.uuid4()}"
    # Store job initial info (in-memory for mock, use DB/Redis in production)
    video_jobs[job_id] = {
        "status": VideoGenerationJobStatus.PENDING,
        "requested_at": time.time(), # Using time.time() for simple timestamp
        "user_id": request_data.user_id,
        "instructor": request_data.instructor_character.value,
        "error_message": None,
        "video_url": None
    }

    # Add the long-running task to background
    background_tasks.add_task(process_video_generation_task, job_id, request_data)

    # Estimate completion time (very rough for mock)
    estimated_time = 5 # minutes
    if "long" in request_data.content_source.raw_text_content if request_data.content_source.raw_text_content else False:
        estimated_time = 10


    return GenerateVideoInitialResponse(
        job_id=job_id,
        status=VideoGenerationJobStatus.PENDING,
        estimated_completion_time_minutes=estimated_time
    )


@app.get("/v1/video-status/{job_id}", response_model=VideoCallbackPayload , summary="Get status of a video generation job")
async def get_video_status_endpoint(job_id: str):
    """
    Allows polling for the status of a video generation job.
    (Alternative to webhook callback for frontend to check status)
    """
    job_info = video_jobs.get(job_id)
    if not job_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found.")
    
    # Construct the response from job_info, matching VideoCallbackPayload
    return VideoCallbackPayload(
        job_id=job_id,
        status=job_info.get("status", VideoGenerationJobStatus.FAILED), # Default to FAILED if status somehow missing
        video_url=job_info.get("video_url"),
        thumbnail_url=job_info.get("thumbnail_url"),
        error_message=job_info.get("error_message"),
        video_duration_seconds=job_info.get("video_duration_seconds")
    )


# To run this FastAPI app locally (from within uplas-ai-agents/ttv_agent/ directory):
# uvicorn main:app --reload --port 8003
