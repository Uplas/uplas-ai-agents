import httpx
import os
import json
from typing import Dict, Optional, Any

# Hypothetical base URL for the third-party avatar service
AVATAR_SERVICE_BASE_URL = os.getenv("AVATAR_SERVICE_BASE_URL", "https://api.avatarservice.xyz/v1")
AVATAR_SERVICE_API_KEY = os.getenv("AVATAR_SERVICE_API_KEY")

# --- Character Config Loading ---
BASE_CHARACTER_ASSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "character_assets")

def load_character_config(instructor_character_name: str) -> Optional[Dict[str, Any]]:
    config_path = os.path.join(BASE_CHARACTER_ASSETS_PATH, instructor_character_name, f"{instructor_character_name}_config.json")
    if not os.path.exists(config_path):
        print(f"Warning: Character config not found for {instructor_character_name} at {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading character config for {instructor_character_name}: {e}")
        return None

class AvatarServiceClient:
    def __init__(self):
        if not AVATAR_SERVICE_API_KEY:
            print("Warning: AVATAR_SERVICE_API_KEY is not set. Avatar service calls will fail.")
        self.headers = {"Authorization": f"Bearer {AVATAR_SERVICE_API_KEY}", "Content-Type": "application/json"}

    async def submit_video_generation_job(
        self,
        script_text: Optional[str] = None, # Some services take script, others audio URL
        audio_gcs_url: Optional[str] = None,
        instructor_character_name: str, # e.g., "susan", "uncle_trevor"
        attire_name: Optional[str] = None, # e.g., "professional_suit_blue"
        language_code: str = "en-US" # For TTS within avatar service if applicable
    ) -> Dict[str, Any]:
        """
        Submits a job to the third-party avatar video generation service.
        Returns a dictionary containing a job ID from the service and its initial status.
        """
        character_config = load_character_config(instructor_character_name)
        if not character_config:
            raise ValueError(f"Invalid instructor character name: {instructor_character_name} or config missing.")

        avatar_id = character_config.get("avatar_id")
        
        # Select attire
        final_attire_id = None
        if attire_name:
            for attire_conf in character_config.get("attires", []):
                if attire_conf.get("name") == attire_name:
                    final_attire_id = attire_conf.get("service_attire_id")
                    break
        if not final_attire_id: # Fallback to default attire
            default_attire_name = character_config.get("default_attire")
            for attire_conf in character_config.get("attires", []):
                if attire_conf.get("name") == default_attire_name:
                    final_attire_id = attire_conf.get("service_attire_id")
                    break
        
        if not avatar_id:
            raise ValueError(f"Avatar ID not found for {instructor_character_name}.")

        payload = {
            "avatar_id": avatar_id,
            "output_format": "mp4", # Or as per service options
            # "language": language_code, # If service does its own TTS
        }
        if final_attire_id:
            payload["attire_id"] = final_attire_id
        
        if audio_gcs_url:
            payload["audio_url"] = audio_gcs_url # Service fetches audio from this GCS URL
        elif script_text:
            payload["script"] = script_text # Service uses its internal TTS
            # May need to specify voice if service does TTS
            voice_id = character_config.get("voice_id_map", {}).get(language_code)
            if voice_id:
                 payload["voice_id"] = voice_id
        else:
            raise ValueError("Either script_text or audio_gcs_url must be provided.")

        print(f"MockAvatarService: Submitting job with payload: {payload}")
        # In a real scenario:
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(f"{AVATAR_SERVICE_BASE_URL}/videos/generate", json=payload, headers=self.headers)
        #     response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
        #     service_response = response.json()
        #     return {"service_job_id": service_response.get("jobId"), "service_status": service_response.get("status")}

        # Mocked response:
        mock_service_job_id = f"avatarsvc_{uuid.uuid4()}"
        print(f"MockAvatarService: Job submitted. Service Job ID: {mock_service_job_id}")
        return {"service_job_id": mock_service_job_id, "service_status": "queued"}


    async def get_video_job_status(self, service_job_id: str) -> Dict[str, Any]:
        """
        Polls the third-party service for the status of a video generation job.
        Returns a dictionary with status, video_url, thumbnail_url, error_message.
        """
        print(f"MockAvatarService: Checking status for Service Job ID: {service_job_id}")
        # In a real scenario:
        # async with httpx.AsyncClient() as client:
        #     response = await client.get(f"{AVATAR_SERVICE_BASE_URL}/videos/status/{service_job_id}", headers=self.headers)
        #     response.raise_for_status()
        #     service_response = response.json()
        #     return {
        #         "service_status": service_response.get("status"), # e.g., "processing", "completed", "failed"
        #         "video_url": service_response.get("outputVideoUrl"),
        #         "thumbnail_url": service_response.get("outputThumbnailUrl"),
        #         "error_message": service_response.get("errorMessage")
        #     }

        # Mocked polling logic:
        # Simulate it takes some time and then completes or fails randomly
        await asyncio.sleep(random.uniform(1, 3)) # Simulate network delay for polling
        
        # This part needs to be more sophisticated if the main TTV agent polls this.
        # For now, let's assume the mock background task in main.py simulates the completion.
        # This method is more for if the TTV agent itself needs to poll the *actual* 3rd party service.
        # Let's have it return a plausible "completed" state for testing the client itself.
        if "error" in service_job_id.lower(): # Simulate an error based on job ID for testing
            return {
                "service_status": "failed",
                "video_url": None,
                "thumbnail_url": None,
                "error_message": "Mocked avatar generation failure."
            }
        
        return {
            "service_status": "completed",
            "video_url": f"https://mockstorage.googleapis.com/{MOCKED_GCS_VIDEO_BUCKET}/final_videos/{service_job_id}.mp4",
            "thumbnail_url": f"https://mockstorage.googleapis.com/{MOCKED_GCS_VIDEO_BUCKET}/thumbnails/{service_job_id}.jpg",
            "error_message": None
        }

# This import is needed for the polling simulation if you uncomment asyncio.sleep
import asyncio
