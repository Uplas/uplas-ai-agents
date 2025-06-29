import logging
import time
import uuid

class MockAvatarAPIClient:
    """A mock client to simulate interactions with a third-party avatar API."""
    def __init__(self, api_key: str):
        if not api_key:
            logging.warning("API key is missing, but mock client will proceed.")
        self.api_key = api_key
        self.jobs = {}

    def submit_video_creation_job(self, text: str, voice_url: str) -> str:
        """Mocks submitting a video creation job and returns a mock job ID."""
        logging.info(f"Mock Submit Job: Received text '{text}' and voice URL '{voice_url}'")
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {"status": "pending", "start_time": time.time()}
        logging.info(f"Mock job created with ID: {job_id}")
        return job_id

    def poll_video_job_status(self, job_id: str) -> dict:
        """Mocks polling for the status of a video creation job."""
        if job_id not in self.jobs:
            logging.error(f"Job ID {job_id} not found in mock job store.")
            return {"status": "error", "message": "Job not found"}
        job = self.jobs[job_id]
        if time.time() - job["start_time"] < 5:
            logging.info(f"Polling Job {job_id}: Status is 'processing'")
            return {"status": "processing"}
        else:
            logging.info(f"Polling Job {job_id}: Status is 'completed'")
            mock_video_url = f"http://example.com/videos/{job_id}.mp4"
            job["status"] = "completed"
            job["video_url"] = mock_video_url
            return {"status": "completed", "video_url": mock_video_url}
