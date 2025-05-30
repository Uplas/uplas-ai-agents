# uplas-ai-agents/ttv_agent/animation_logic/avatar_api_client.py
import httpx
import os
import uuid
import asyncio
import random
import logging
from typing import Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Environment variables for the third-party avatar service
# These MUST be set in your deployment environment.
THIRD_PARTY_AVATAR_API_KEY_ENV = "THIRD_PARTY_AVATAR_API_KEY"
THIRD_PARTY_AVATAR_BASE_URL_ENV = "THIRD_PARTY_AVATAR_BASE_URL"

class AvatarJobError(Exception):
    """Custom exception for avatar video job failures."""
    pass

class ThirdPartyAvatarAPIClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, is_mock: bool = False):
        """
        Initializes the client for the third-party avatar service.

        Args:
            api_key: The API key for the avatar service.
            base_url: The base URL for the avatar service API.
            is_mock: If True, the client will use mocked responses. Useful for development
                     if API key/base_url are not provided or for testing.
        """
        self.api_key = api_key or os.getenv(THIRD_PARTY_AVATAR_API_KEY_ENV)
        self.base_url = base_url or os.getenv(THIRD_PARTY_AVATAR_BASE_URL_ENV)
        self.is_mock = is_mock

        if not self.is_mock and (not self.api_key or not self.base_url):
            logger.warning(
                f"{THIRD_PARTY_AVATAR_API_KEY_ENV} or {THIRD_PARTY_AVATAR_BASE_URL_ENV} are not set. "
                "Real API calls will likely fail. Consider running in mock mode or setting these environment variables."
            )
            # You might choose to force mock mode if critical configs are missing:
            # self.is_mock = True
            # Or raise an error:
            # raise ValueError("API key and base URL are required for non-mock mode.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.timeout = httpx.Timeout(60.0, connect=10.0) # Increased timeout for potentially long operations

        # Mock state for polling (if is_mock is True)
        self._mock_jobs: Dict[str, Dict[str, Any]] = {}

    async def submit_video_creation_job(
        self,
        service_avatar_id: str,                 # ID of the avatar in the third-party service
        audio_file_gcs_url: str,                # GCS URL of the pre-generated audio
        service_attire_id: Optional[str] = None,
        language_code: Optional[str] = None,    # e.g., "en-US", important for lip-sync accuracy
        background_settings: Optional[Dict] = None, # e.g., {"color": "#FFFFFF", "image_url": "..."}
        output_webhook_url: Optional[str] = None, # URL for the service to notify upon completion
        custom_metadata: Optional[Dict[str, str]] = None # Any other parameters the service might need
    ) -> Dict[str, Any]:
        """
        Submits a request to the third-party API to create an avatar video.

        Args:
            service_avatar_id: The unique identifier for the character model on the avatar platform.
            audio_file_gcs_url: Publicly accessible GCS URL to the audio file for lip-syncing.
            service_attire_id: (Optional) Identifier for the character's attire/outfit.
            language_code: (Optional) BCP-47 language code of the audio, crucial for accurate lip-sync.
            background_settings: (Optional) Dictionary defining video background (e.g., color, image URL).
            output_webhook_url: (Optional) Webhook URL for the avatar service to call upon job completion/failure.
            custom_metadata: (Optional) Any additional custom parameters required by the specific avatar service.

        Returns:
            A dictionary containing at least 'service_job_id' (the ID assigned by the third-party service)
            and 'initial_status' (e.g., 'queued', 'processing').
            Example: {"service_job_id": "xyz789", "initial_status": "queued"}
        """
        if self.is_mock:
            return await self._mock_submit_video_creation_job(service_avatar_id, audio_file_gcs_url, service_attire_id)

        if not self.api_key or not self.base_url:
            raise AvatarJobError("API key or base URL for avatar service is not configured.")

        # --- CONSTRUCT THE PAYLOAD BASED ON THE ACTUAL THIRD-PARTY SERVICE'S API DOCUMENTATION ---
        # This is a generic placeholder. You MUST replace this with the specific payload structure
        # required by your chosen avatar video generation service.
        payload = {
            "avatarId": service_avatar_id,
            "audioUrl": audio_file_gcs_url,
            "outputSettings": {"format": "mp4", "resolution": "1080p"}, # Example
        }
        if service_attire_id:
            # How attire is specified varies greatly (e.g., part of avatarId, separate param, in customization object)
            payload.setdefault("customization", {})["attireId"] = service_attire_id
        if language_code:
            # Language might be part of voice settings or a top-level parameter for lip-sync
            payload.setdefault("source", {})["language"] = language_code
        if background_settings:
            payload.setdefault("customization", {})["background"] = background_settings
        if output_webhook_url:
            payload["webhookUrl"] = output_webhook_url
        if custom_metadata:
            payload.update(custom_metadata) # Or merge into a specific field like 'metadata'

        api_endpoint = f"{self.base_url}/videos" # Example endpoint, adjust as per service docs

        logger.info(f"Submitting video creation job to {api_endpoint} for avatar {service_avatar_id}.")
        logger.debug(f"Payload (sample): {str(payload)[:200]}...") # Log a sample, be careful with sensitive data

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(api_endpoint, json=payload, headers=self.headers)
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
                response_data = response.json()

                service_job_id = response_data.get("jobId") # Or "id", "videoId", etc. - check service docs
                initial_status = response_data.get("status", "unknown") # e.g., "queued", "processing"

                if not service_job_id:
                    logger.error(f"Avatar service did not return a job ID. Response: {response_data}")
                    raise AvatarJobError(f"Avatar service did not return a job ID. Response: {response_data}")

                logger.info(f"Video creation job submitted successfully. Service Job ID: {service_job_id}, Initial Status: {initial_status}")
                return {"service_job_id": service_job_id, "initial_status": initial_status}

            except httpx.HTTPStatusError as e:
                err_msg = f"Avatar API HTTP Error ({e.response.status_code}) submitting job: {e.response.text}"
                logger.error(err_msg, exc_info=True)
                raise AvatarJobError(err_msg)
            except Exception as e:
                err_msg = f"Avatar API Request Error submitting job: {str(e)}"
                logger.error(err_msg, exc_info=True)
                raise AvatarJobError(err_msg)

    async def poll_video_job_status(self, service_job_id: str) -> Dict[str, Any]:
        """
        Polls the third-party API for the status of a video generation job.

        Args:
            service_job_id: The job ID received from the avatar service.

        Returns:
            A dictionary including the job's current 'status', and if completed,
            'videoUrl', 'thumbnailUrl', 'durationSeconds'. If failed, includes 'errorMessage'.
            Example success:
            {
                "status": "completed",
                "videoUrl": "https://...",
                "thumbnailUrl": "https://...",
                "durationSeconds": 180.5
            }
            Example failure:
            {
                "status": "failed",
                "errorMessage": "Audio processing error."
            }
            Example processing:
            {
                "status": "processing",
                "progressPercent": 50 # Optional, if provided by service
            }
        """
        if self.is_mock:
            return await self._mock_poll_video_job_status(service_job_id)

        if not self.api_key or not self.base_url:
            raise AvatarJobError("API key or base URL for avatar service is not configured.")

        api_endpoint = f"{self.base_url}/videos/{service_job_id}/status" # Example, adjust as per service docs
        logger.info(f"Polling job status from {api_endpoint} for service job ID: {service_job_id}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(api_endpoint, headers=self.headers)
                response.raise_for_status()
                response_data = response.json()

                # --- PARSE THE RESPONSE DATA BASED ON THE ACTUAL THIRD-PARTY SERVICE'S API DOCUMENTATION ---
                # This is a generic placeholder. You MUST adapt this to the service's response structure.
                status = response_data.get("status", "unknown") # e.g., "queued", "processing", "completed", "failed"
                video_url = response_data.get("outputVideoUrl") or response_data.get("video_url")
                thumbnail_url = response_data.get("outputThumbnailUrl") or response_data.get("thumbnail_url")
                duration = response_data.get("duration") or response_data.get("durationSeconds")
                error_message = response_data.get("error") or response_data.get("errorMessage")
                progress = response_data.get("progress") # Optional progress percentage

                logger.info(f"Status for job {service_job_id}: {status}. Video URL: {video_url if video_url else 'N/A'}")

                return {
                    "status": status.lower(), # Standardize to lowercase
                    "videoUrl": video_url,
                    "thumbnailUrl": thumbnail_url,
                    "durationSeconds": float(duration) if duration is not None else None,
                    "errorMessage": error_message,
                    "progressPercent": int(progress) if progress is not None else None
                }

            except httpx.HTTPStatusError as e:
                err_msg = f"Avatar API HTTP Error ({e.response.status_code}) polling job {service_job_id}: {e.response.text}"
                logger.error(err_msg, exc_info=True)
                # Depending on the error, you might return a 'failed' status or re-raise
                return {"status": "failed", "errorMessage": f"API error polling status: {e.response.status_code}"}
            except Exception as e:
                err_msg = f"Avatar API Request Error polling job {service_job_id}: {str(e)}"
                logger.error(err_msg, exc_info=True)
                return {"status": "failed", "errorMessage": f"Client error polling status: {str(e)}"}


    # --- Mock Methods ---
    async def _mock_submit_video_creation_job(
        self, service_avatar_id: str, audio_file_gcs_url: str, service_attire_id: Optional[str]
    ) -> Dict[str, Any]:
        await asyncio.sleep(random.uniform(0.1, 0.3)) # Simulate network latency
        mock_service_job_id = f"mock_avatarsvc_job_{uuid.uuid4().hex[:12]}"
        
        self._mock_jobs[mock_service_job_id] = {
            "status": "queued", # Initial status
            "avatar_id": service_avatar_id,
            "audio_url": audio_file_gcs_url,
            "attire_id": service_attire_id,
            "submission_time": time.time(),
            "render_start_time": None,
            "completion_time": None,
            "poll_count": 0
        }
        logger.info(f"MOCK AvatarClient: Job submitted successfully. Service Job ID: {mock_service_job_id}")
        return {"service_job_id": mock_service_job_id, "initial_status": "queued"}

    async def _mock_poll_video_job_status(self, service_job_id: str) -> Dict[str, Any]:
        await asyncio.sleep(random.uniform(0.5, 1.5)) # Simulate polling network delay
        
        job_data = self._mock_jobs.get(service_job_id)
        if not job_data:
            logger.error(f"MOCK AvatarClient: Polled for unknown job ID {service_job_id}")
            return {"status": "failed", "errorMessage": "Mock job ID not found."}

        job_data["poll_count"] += 1
        current_time = time.time()

        # Simulate progression
        if job_data["status"] == "queued":
            if (current_time - job_data["submission_time"]) > 2: # After 2s, move to processing
                job_data["status"] = "processing"
                job_data["render_start_time"] = current_time
                logger.info(f"MOCK AvatarClient: Job {service_job_id} moved to 'processing'.")

        elif job_data["status"] == "processing":
            # Simulate random failure for testing
            if "fail_mock_job" in service_job_id and job_data["poll_count"] > 2 : # Test condition
                job_data["status"] = "failed"
                job_data["errorMessage"] = "Mocked explicit failure during polling."
                logger.info(f"MOCK AvatarClient: Job {service_job_id} moved to 'failed' (simulated).")
            elif job_data["render_start_time"] and (current_time - job_data["render_start_time"]) > random.uniform(5, 15): # Simulate 5-15s render time
                job_data["status"] = "completed"
                job_data["completion_time"] = current_time
                job_data["videoUrl"] = f"https://mock.storage.uplas.com/generated_videos/{service_job_id}.mp4"
                job_data["thumbnailUrl"] = f"https://mock.storage.uplas.com/generated_videos/thumbs/{service_job_id}.jpg"
                job_data["durationSeconds"] = random.uniform(120, 360) # 2 to 6 minutes
                logger.info(f"MOCK AvatarClient: Job {service_job_id} moved to 'completed'.")
        
        # Return current state
        response = {"status": job_data["status"]}
        if job_data["status"] == "completed":
            response.update({
                "videoUrl": job_data["videoUrl"],
                "thumbnailUrl": job_data["thumbnailUrl"],
                "durationSeconds": job_data["durationSeconds"]
            })
        elif job_data["status"] == "failed":
            response["errorMessage"] = job_data.get("errorMessage", "Mocked unknown failure.")
        elif job_data["status"] == "processing":
            # Simulate progress
            if job_data["render_start_time"]:
                elapsed_render_time = current_time - job_data["render_start_time"]
                # Assume total render time is ~10s for mock
                progress = min(int((elapsed_render_time / 10.0) * 100), 99) if elapsed_render_time < 10 else 99
                response["progressPercent"] = progress


        logger.debug(f"MOCK AvatarClient: Polling job {service_job_id}. Current status: {job_data['status']}")
        return response

