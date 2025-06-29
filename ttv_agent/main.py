import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import textwrap
import uuid
import requests

# --- Local Imports ---
from animation_logic.avatar_api_client import MockAvatarAPIClient

load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# --- Environment Variable Configuration ---
AI_TUTOR_AGENT_URL = os.getenv("AI_TUTOR_AGENT_URL", "http://localhost:8001")
TTS_AGENT_URL = os.getenv("TTS_AGENT_URL", "http://localhost:8002")
AVATAR_API_KEY = os.getenv("AVATAR_API_KEY", "mock_api_key_for_local_dev")

avatar_client = MockAvatarAPIClient(api_key=AVATAR_API_KEY)
OUTPUT_DIR = "local_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _generate_text_slide_image(text: str, width: int = 1920, height: int = 1080) -> str:
    """Generates an image with the given text using Pillow."""
    logging.info("Generating text slide image...")
    img = Image.new('RGB', (width, height), color=(26, 26, 26))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except IOError:
        logging.warning("Arial font not found. Using default PIL font.")
        font = ImageFont.load_default()
    margin, offset = 80, 100
    wrapper = textwrap.TextWrapper(width=40)
    lines = wrapper.wrap(text=text)
    for line in lines:
        d.text((margin, offset), line, font=font, fill=(255, 255, 255))
        offset += font.getbbox(line)[3] + 10
    image_filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join(OUTPUT_DIR, image_filename)
    img.save(image_path)
    logging.info(f"Saved slide image to {image_path}")
    return image_path

class VideoRequest(BaseModel):
    text_content: str
    mode: str

@app.post("/generate_video")
def generate_video(request: VideoRequest):
    """Generates a video using either the avatar or slideshow mode."""
    logging.info(f"Received video generation request in '{request.mode}' mode.")
    if request.mode == 'SLIDESHOW_MVP':
        try:
            sentences = request.text_content.split('. ')
            image_paths = [_generate_text_slide_image(sentence) for sentence in sentences if sentence]
            logging.info("Slideshow images generated. Video creation step is mocked.")
            return {
                "status": "success",
                "message": "Slideshow images generated successfully.",
                "image_paths": image_paths,
                "note": "Video compilation is a manual step in local dev."
            }
        except Exception as e:
            logging.error(f"Error in SLIDESHOW_MVP mode: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate slideshow.")
    else:
        raise HTTPException(status_code=400, detail="Invalid mode specified.")

@app.get("/")
def read_root():
    return {"message": "TTV Agent is running in local mode."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
