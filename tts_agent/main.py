import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# --- Mock GCS and TTS ---
# This setup simulates cloud storage and text-to-speech for local development.
MOCK_GCS_BUCKET_DIR = "local_mock_gcs"
os.makedirs(MOCK_GCS_BUCKET_DIR, exist_ok=True)

def mock_synthesize_speech(text: str) -> bytes:
    """Mocks the Text-to-Speech API by returning sample audio data."""
    logging.info(f"Mocking speech synthesis for text: '{text}'")
    return f"mock audio for: {text}".encode('utf-8')

def mock_upload_to_gcs(bucket_name: str, destination_blob_name: str, contents: bytes) -> str:
    """Mocks uploading a file to GCS by saving it to the local filesystem."""
    file_path = os.path.join(MOCK_GCS_BUCKET_DIR, destination_blob_name)
    with open(file_path, "wb") as f:
        f.write(contents)
    logging.info(f"Mock GCS: Saved file to {file_path}")
    return f"file://{os.path.abspath(file_path)}"

class TTSRequest(BaseModel):
    text: str
    gcs_bucket_name: str
    output_gcs_path: str

@app.post("/synthesize_speech")
def synthesize_speech_endpoint(request: TTSRequest):
    """Generates speech from text and 'uploads' it to a local directory."""
    try:
        logging.info(f"Received TTS request for text: '{request.text}'")
        audio_content = mock_synthesize_speech(request.text)
        unique_filename = f"{request.output_gcs_path}/{uuid.uuid4()}.mp3"
        public_url = mock_upload_to_gcs(
            bucket_name=request.gcs_bucket_name,
            destination_blob_name=unique_filename,
            contents=audio_content
        )
        logging.info(f"Generated mock audio and saved to: {public_url}")
        return {"gcs_path": public_url}
    except Exception as e:
        logging.error(f"Error in mock speech synthesis: {e}")
        raise HTTPException(status_code=500, detail="Failed to synthesize speech.")

@app.get("/")
def read_root():
    return {"message": "TTS Agent is running in local mock mode."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
