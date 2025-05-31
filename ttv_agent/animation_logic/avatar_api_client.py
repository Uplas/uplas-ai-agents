# uplas-ai-agents/ttv_agent/animation_logic/avatar_api_client.py
import httpx  # Using httpx for asynchronous HTTP requests
import os
import uuid
import asyncio
import random
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# Environment variables for YOUR CHOSEN third-party avatar service
# These MUST be set in your deployment environment for non-mock operation.
THIRD_PARTY_AVATAR_API_KEY_ENV = "THIRD_PARTY_AVATAR_API_KEY"
THIRD_PARTY_AVATAR_BASE_URL_ENV = "THIRD_PARTY_AVATAR_BASE_URL"  # e.g., "https://api.youravatarservice.com/v2"
# InnovateAI: Optional environment variable to force mock mode for safety
FORCE_MOCK_MODE_ENV = "TTV_AVATAR_CLIENT_FORCE_MOCK_MODE"

class AvatarJobError(Exception):
    """Custom exception for errors related to the avatar video generation service."""
    def __init__(self, message: str, status_code: Optional[int] = None, service_error_details: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.service_error_details = service_error_details

    def __str__(self):
        return f"{super().__str__()} (Status Code: {self.status_code}, Service Details: {self.service_error_details})"


class ThirdPartyAvatarAPIClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, is_mock_override: Optional[bool] = None):
        """
        Initializes the client for the third-party avatar video generation service.

        Args:
            api_key: The API key for the avatar service. If None, reads from environment.
            base_url: The base URL for the avatar service API. If None, reads from environment.
            is_mock_override: Explicitly set mock mode, bypassing environment variable check for forcing mock.
        """
        self.api_key = api_key or os.getenv(THIRD_PARTY_AVATAR_API_KEY_ENV)
        self.base_url = base_url or os.getenv(THIRD_PARTY_AVATAR_BASE_URL_ENV)
        
        force_mock_env = os.getenv(FORCE_MOCK_MODE_ENV, "false").lower() == "true"
        self.is_mock_mode = is_mock_override if is_mock_override is not None else force_mock_env

        if not self.is_mock_mode and (not self.api_key or not self.base_url):
            warning_msg = (
                f"NovaSpark Critical Warning: {THIRD_PARTY_AVATAR_API_KEY_ENV} or "
                f"{THIRD_PARTY_AVATAR_BASE_URL_ENV} are not set. Real API calls to the "
                "third-party avatar service WILL FAIL. Forcing client into MOCK MODE. "
                "To enable real calls, set these environment variables and ensure "
                f"'{FORCE_MOCK_MODE_ENV}' is not 'true'."
            )
            logger.error(warning_msg)
            self.is_mock_mode = True # Force mock mode if critical configs are missing for real operation

        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        # --- NovaSpark/Mugambi Action: Configure Authentication Headers ---
        # This is HIGHLY service-specific. CONSULT YOUR CHOSEN SERVICE'S API DOCUMENTATION.
        # Choose ONE of the common methods below or implement your service's specific scheme.
        if self.api_key:
            # Option 1: Bearer Token
            # self.headers["Authorization"] = f"Bearer {self.api_key}"
            # logger.info("NovaSpark Info: Using Bearer Token authentication for avatar service.")

            # Option 2: Custom API Key Header (e.g., X-Api-Key)
            # self.headers["X-Api-Key"] = self.api_key # FIXME: Replace X-Api-Key with actual header name
            # logger.info("NovaSpark Info: Using custom API key header for avatar service.")
            
            # Option 3: API key in every request (some services might require this differently,
            # e.g. as part of the JSON payload or query params for GETs)
            # This will be handled in submit/poll methods if needed.
            logger.warning("NovaSpark Reminder: API key is available, but AUTHENTICATION HEADER IS NOT EXPLICITLY SET UP. Mugambi, please configure based on your chosen service in `__init__`.")
            self.headers["X-TEMP-API-KEY-PLACEHOLDER"] = self.api_key # Temporary, so it's used somewhere
        elif not self.is_mock_mode:
            logger.error("NovaSpark Critical Error: Avatar client is in REAL mode but API key is MISSING. Calls will likely fail.")
        else: # Mock mode and no API key
            logger.info("NovaSpark Info: Avatar client in MOCK mode, API key not configured (as expected for mock).")


        self.timeout_config = httpx.Timeout(300.0, connect=60.0) # Increased: 5 min total, 1 min connect

        self._mock_jobs: Dict[str, Dict[str, Any]] = {}
        if self.is_mock_mode:
            logger.info("NovaSpark: ThirdPartyAvatarAPIClient initialized in MOCK MODE. No actual API calls will be made to external services.")

    async def _map_background_to_service_format(self, background_settings: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        NovaSpark/Mugambi Helper: Maps Uplas's generic background_settings to the
        specific format required by YOUR CHOSEN third-party avatar service.
        This is a placeholder and needs to be implemented based on service documentation.
        """
        if not background_settings:
            return None

        # FIXME: Mugambi, implement the actual mapping logic here.
        # Example conceptual mapping:
        # service_bg_payload = {}
        # bg_type = background_settings.get("type")
        # if bg_type == "image_url" and "url" in background_settings:
        #     service_bg_payload["service_specific_image_field_name"] = background_settings["url"]
        # elif bg_type == "solid_color" and "hex" in background_settings:
        #     service_bg_payload["service_specific_color_field_name"] = background_settings["hex"]
        # elif bg_type == "preset_scene_id" and "id" in background_settings:
        #     service_bg_payload["service_specific_scene_id_field"] = background_settings["id"]
        # # ... add other mappings as needed by your service
        #
        # if service_bg_payload:
        #    logger.debug(f"NovaSpark: Mapped background settings to service format: {service_bg_payload}")
        #    return service_bg_payload
        
        logger.warning(f"NovaSpark Placeholder: Background mapping not fully implemented. Passing raw settings: {background_settings}. Mugambi, please implement `_map_background_to_service_format`.")
        return {"service_specific_background_placeholder": background_settings} # Placeholder


    async def submit_video_creation_job(
        self,
        service_avatar_id: str,
        audio_file_gcs_url: str, # Ensure this is publicly accessible or a signed URL if service requires
        script_text_for_lipsync_data: Optional[str] = None, # Some services might prefer raw text for better lipsync data along with audio
        service_attire_id: Optional[str] = None,
        language_code: Optional[str] = None, 
        background_settings: Optional[Dict[str, Any]] = None,
        output_webhook_url: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        NovaSpark Refined: Submits a video creation request to the CHOSEN third-party avatar service.
        Mugambi, this method requires YOU to implement the service-specific API call logic.
        """
        if self.is_mock_mode:
            return await self._mock_submit_video_creation_job(service_avatar_id, audio_file_gcs_url, service_attire_id, custom_metadata)

        if not self.base_url: # API key checked in init
            err_msg = "NovaSpark Error: Base URL for avatar service is not configured for non-mock operation."
            logger.error(err_msg)
            raise AvatarJobError(err_msg)

        # --- NovaSpark/Mugambi Action: Implement YOUR SERVICE'S submit logic below ---
        # FIXME: Replace with your service's actual endpoint for submitting video jobs
        api_endpoint = f"{self.base_url}/videos" 
        # api_endpoint = f"{self.base_url}/generations" # Another common pattern

        # FIXME: Construct the precise JSON payload required by YOUR CHOSEN SERVICE.
        # This is THE MOST CRITICAL part and is entirely service-dependent.
        payload: Dict[str, Any] = {
            # --- Common Mandatory Fields (Highly service-specific field names) ---
            "avatar_id": service_avatar_id, # FIXME: e.g., "characterId", "model_id", "presenter_id"
            # Audio Input: Services handle this very differently
            # Option 1: URL to pre-generated audio (preferred for Uplas, as TTS is separate)
            "audio_url": audio_file_gcs_url, # FIXME: e.g., "audioSourceUrl", "speech_url"
            # Option 2: Service does its own TTS from script (less ideal for Uplas if we want our own TTS control)
            # "script_text": script_text_for_lipsync_data, # FIXME: e.g., "text_to_speak"
            # "voice_id": "service_specific_voice_id_for_avatar", # FIXME: if service does TTS
            
            # --- Common Optional Fields (Map from our function args) ---
            # "attire_id": service_attire_id, # FIXME: e.g., "outfitId", "clothing_style" (Dynamic Attire)
            # "language": language_code, # FIXME: For lipsync accuracy or service-side TTS language hint
            # "background": await self._map_background_to_service_format(background_settings), # FIXME (using helper)
            # "callback_url": output_webhook_url, # FIXME: e.g., "notificationUrl", "webhook"
            # "metadata": custom_metadata, # FIXME: e.g., "custom_tags", "passthrough_data"
            
            # --- Other Potential Service-Specific Parameters ---
            # "output_format": "mp4", # FIXME: e.g., "video_codec", "container"
            # "resolution": "1080p", # FIXME: e.g., "1920x1080", "video_quality"
            # "enable_lipsync_enhancement": True, # FIXME: If a specific flag for lipsync quality exists
        }
        
        # Refine payload: remove None values if your service API is strict or add specific optionals
        payload = {k: v for k, v in payload.items() if v is not None} 
        if service_attire_id: payload["attire_id_placeholder"] = service_attire_id # FIXME
        if language_code: payload["language_placeholder"] = language_code # FIXME
        # ... and so on for other optional fields based on service docs.

        logger.info(f"NovaSpark (REAL AvatarClient): Submitting video creation job to {api_endpoint} for avatar '{service_avatar_id}'.")
        logger.debug(f"NovaSpark (REAL AvatarClient): Service Payload (first 500 chars): {str(payload)[:500]}...")

        async with httpx.AsyncClient(timeout=self.timeout_config, follow_redirects=True) as client:
            try:
                response = await client.post(api_endpoint, json=payload, headers=self.headers)
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
                response_data = response.json()

                # FIXME: Extract the job ID and initial status from YOUR SERVICE'S response.
                # These field names WILL VARY SIGNIFICANTLY.
                service_job_id = response_data.get("id") # Common pattern
                # service_job_id = response_data.get("job", {}).get("id") # Another common pattern
                
                initial_status_from_service = response_data.get("status", "unknown") # Common pattern
                # initial_status_from_service = response_data.get("state", "unknown") # Another common pattern

                if not service_job_id:
                    error_detail = f"Avatar service response did not contain a job ID. Response: {str(response_data)[:1000]}"
                    logger.error(f"NovaSpark (REAL AvatarClient): {error_detail}")
                    raise AvatarJobError(error_detail, status_code=response.status_code, service_error_details=response_data)

                logger.info(f"NovaSpark (REAL AvatarClient): Job submitted. Service Job ID: {service_job_id}, Initial Service Status: {initial_status_from_service}")
                return {"service_job_id": str(service_job_id), "initial_status": self._standardize_service_status(initial_status_from_service)}

            except httpx.HTTPStatusError as e:
                err_content = e.response.text[:500] if e.response else "N/A"
                err_msg = f"HTTP Error ({e.response.status_code}) submitting job to avatar service: {err_content}"
                logger.error(f"NovaSpark (REAL AvatarClient): {err_msg}", exc_info=True)
                raise AvatarJobError(err_msg, status_code=e.response.status_code, service_error_details=err_content) from e
            except (httpx.RequestError, json.JSONDecodeError) as e: # More specific non-HTTP errors
                err_msg = f"Request or JSON parsing error submitting job to avatar service: {str(e)}"
                logger.error(f"NovaSpark (REAL AvatarClient): {err_msg}", exc_info=True)
                raise AvatarJobError(err_msg) from e
        # --- END NovaSpark/Mugambi Action ---

    async def poll_video_job_status(self, service_job_id: str) -> Dict[str, Any]:
        """
        NovaSpark Refined: Polls the CHOSEN third-party API for video job status.
        Mugambi, this method requires YOU to implement the service-specific API call and response parsing.
        """
        if self.is_mock_mode:
            return await self._mock_poll_video_job_status(service_job_id)

        if not self.base_url:
            err_msg = "NovaSpark Error: Base URL for avatar service is not configured for non-mock operation."
            logger.error(err_msg)
            raise AvatarJobError(err_msg)

        # --- NovaSpark/Mugambi Action: Implement YOUR SERVICE'S poll logic below ---
        # FIXME: Replace with your service's actual endpoint for polling job status
        # api_endpoint = f"{self.base_url}/videos/{service_job_id}/status" 
        api_endpoint = f"{self.base_url}/generations/{service_job_id}" # Common pattern

        logger.debug(f"NovaSpark (REAL AvatarClient): Polling job status from {api_endpoint} for service job ID: {service_job_id}")

        async with httpx.AsyncClient(timeout=self.timeout_config, follow_redirects=True) as client:
            try:
                response = await client.get(api_endpoint, headers=self.headers)
                response.raise_for_status()
                response_data = response.json()

                # FIXME: Parse status and other relevant fields from YOUR SERVICE'S response.
                # Field names WILL VARY. Map to our standardized status values.
                status_from_service = response_data.get("status", "unknown") 
                # status_from_service = response_data.get("state", "unknown")
                job_status = self._standardize_service_status(status_from_service)

                # FIXME: Extract these based on your service's response structure
                video_url = response_data.get("video_url")  # e.g., "output_url", "result.video_link"
                thumbnail_url = response_data.get("thumbnail_url") # e.g., "preview_image_url"
                duration_str = response_data.get("duration") # e.g., in seconds, or "HH:MM:SS"
                error_msg = response_data.get("error_message") # e.g., "failure_reason", "details"
                progress_val = response_data.get("progress") # e.g., 0-100, or text "50%"
                
                # Robust parsing for duration and progress
                duration_seconds: Optional[float] = None
                if duration_str is not None:
                    try: duration_seconds = float(duration_str)
                    except ValueError: logger.warning(f"NovaSpark: Could not parse duration '{duration_str}' for job {service_job_id}.")
                
                progress_percent: Optional[int] = None
                if progress_val is not None:
                    try: progress_percent = int(float(str(progress_val).replace('%','').strip()))
                    except ValueError: logger.warning(f"NovaSpark: Could not parse progress '{progress_val}' for job {service_job_id}.")

                logger.debug(f"NovaSpark (REAL AvatarClient): Parsed status for job {service_job_id}: '{job_status}'. Raw: '{status_from_service}'. Progress: {progress_percent}%.")

                return {
                    "status": job_status,
                    "videoUrl": video_url,
                    "thumbnailUrl": thumbnail_url,
                    "durationSeconds": duration_seconds,
                    "errorMessage": error_msg if job_status == "failed" else None,
                    "progressPercent": progress_percent
                }

            except httpx.HTTPStatusError as e:
                err_content = e.response.text[:500] if e.response else "N/A"
                err_msg = f"HTTP Error ({e.response.status_code}) polling job {service_job_id}: {err_content}"
                logger.error(f"NovaSpark (REAL AvatarClient): {err_msg}", exc_info=True)
                # For polling, it's often better to return a "failed" status for the TTV agent to handle,
                # rather than raising an exception that stops the whole TTV job abruptly if it's a transient poll issue.
                # However, if the error is like 401/403 (auth) or 404 (job not found), it might be non-recoverable.
                if e.response.status_code in [401, 403, 404]: # Non-recoverable by more polling
                     raise AvatarJobError(err_msg, status_code=e.response.status_code, service_error_details=err_content) from e
                return {"status": "failed", "errorMessage": f"API error during status poll: Code {e.response.status_code}. Detail: {err_content}"}
            except (httpx.RequestError, json.JSONDecodeError) as e:
                err_msg = f"Request or JSON parsing error polling job {service_job_id}: {str(e)}"
                logger.error(f"NovaSpark (REAL AvatarClient): {err_msg}", exc_info=True)
                return {"status": "failed", "errorMessage": f"Client-side error during status poll: {str(e)}"}
        # --- END NovaSpark/Mugambi Action ---

    def _standardize_service_status(self, service_status: Optional[str]) -> str:
        """
        NovaSpark Helper: Standardizes status strings from various third-party services.
        Mugambi: You MUST adapt this mapping based on YOUR CHOSEN avatar service.
        """
        if not service_status: return "unknown"
        s = str(service_status).lower().strip()

        # FIXME: Mugambi, add your service's specific status mappings here
        if s in ["succeeded", "finished", "done", "success", "completed", "ready"]: return "completed"
        if s in ["error", "failure", "cancelled", "timeout", "timed_out", "rejected", "aborted"]: return "failed"
        if s in ["submitted", "pending", "waiting", "in_queue", "created", "new"]: return "queued"
        if s in ["in_progress", "active", "working", "generating", "synthesizing", "processing_audio", "preparing_video", "started"]: return "processing"
        if s in ["rendering_video", "rendering", "finalizing"]: return "rendering"
        
        logger.warning(f"NovaSpark: Unmapped service status '{s}' encountered. Defaulting to 'processing'. Update `_standardize_service_status` in avatar_api_client.py.")
        return "processing" 

    # --- Mock Methods (Enhanced for better testing by NovaSpark) ---
    async def _mock_submit_video_creation_job(
        self, service_avatar_id: str, audio_file_gcs_url: str, 
        service_attire_id: Optional[str], custom_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        await asyncio.sleep(random.uniform(0.01, 0.05)) # Faster mock
        mock_service_job_id = f"mock_job_{uuid.uuid4().hex[:6]}"
        
        uplas_job_id = "unknown_uplas_job"
        if custom_metadata and custom_metadata.get("uplas_job_id"):
            uplas_job_id = custom_metadata.get("uplas_job_id")

        self._mock_jobs[mock_service_job_id] = {
            "status": "queued", "submission_time": time.time(), 
            "poll_count": 0, "progressPercent": 0,
            "uplas_job_id": uplas_job_id, # Store for richer mock polling
            "service_avatar_id": service_avatar_id, 
            "audio_url_submitted": audio_file_gcs_url,
            "attire_id_submitted": service_attire_id
        }
        logger.info(f"NovaSpark (MOCK AvatarClient): Job {uplas_job_id} submitted. Mock Service Job ID: {mock_service_job_id}")
        return {"service_job_id": mock_service_job_id, "initial_status": "queued"}

    async def _mock_poll_video_job_status(self, service_job_id: str) -> Dict[str, Any]:
        await asyncio.sleep(random.uniform(0.02, 0.08)) # Simulate network delay
        job_data = self._mock_jobs.get(service_job_id)
        if not job_data:
            logger.error(f"NovaSpark (MOCK AvatarClient): Mock job ID {service_job_id} not found.")
            return {"status": "failed", "errorMessage": "Mock job ID not found."}

        job_data["poll_count"] += 1
        current_time = time.time()
        status_response: Dict[str, Any] = {"status": job_data["status"], "progressPercent": job_data.get("progressPercent", 0)}

        # More deterministic mock state transitions based on poll count for testing different phases
        # These timings are relative to poll counts, not absolute time, for easier test control.
        uplas_job_id = job_data.get("uplas_job_id", service_job_id)

        if "force_fail_mock" in uplas_job_id and job_data["poll_count"] >= 2:
            job_data["status"] = "failed"
            job_data["errorMessage"] = "NovaSpark: Mock forced failure for TTV testing."
            logger.info(f"NovaSpark (MOCK AvatarClient): Job {uplas_job_id} (Service: {service_job_id}) -> failed (forced).")
        elif job_data["status"] == "queued":
            if job_data["poll_count"] >= 2: # After 1 poll in queued, move to processing
                job_data["status"] = "processing"
                job_data["progressPercent"] = random.randint(5, 25)
                logger.info(f"NovaSpark (MOCK AvatarClient): Job {uplas_job_id} (Service: {service_job_id}) -> {job_data['status']} ({job_data['progressPercent']}%)")
        elif job_data["status"] == "processing":
            if job_data["poll_count"] >= 4: # After 2 more polls in processing, move to rendering
                job_data["status"] = "rendering"
                job_data["progressPercent"] = random.randint(60, 85)
                logger.info(f"NovaSpark (MOCK AvatarClient): Job {uplas_job_id} (Service: {service_job_id}) -> {job_data['status']} ({job_data['progressPercent']}%)")
            else:
                job_data["progressPercent"] = min(job_data.get("progressPercent", 0) + random.randint(15,30), 59)
        elif job_data["status"] == "rendering":
            if job_data["poll_count"] >= 6: # After 2 more polls in rendering, complete
                job_data["status"] = "completed"
                job_data["progressPercent"] = 100
                logger.info(f"NovaSpark (MOCK AvatarClient): Job {uplas_job_id} (Service: {service_job_id}) -> completed.")
            else:
                 job_data["progressPercent"] = min(job_data.get("progressPercent", 0) + random.randint(10,20), 99)
        
        status_response["status"] = job_data["status"]
        status_response["progressPercent"] = job_data.get("progressPercent", 0)

        if job_data["status"] == "completed":
            status_response["videoUrl"] = f"https://mock.storage.uplas.com/generated_videos/{uplas_job_id}.mp4"
            status_response["thumbnailUrl"] = f"https://mock.storage.uplas.com/generated_videos/thumbs/{uplas_job_id}.jpg"
            status_response["durationSeconds"] = round(random.uniform(25.0, 185.0), 1)
        elif job_data["status"] == "failed":
            status_response["errorMessage"] = job_data.get("errorMessage", "NovaSpark: Mocked unknown failure.")

        logger.debug(f"NovaSpark (MOCK AvatarClient): Polling job {uplas_job_id} (Service: {service_job_id}). Returning: {status_response}")
        return status_response
