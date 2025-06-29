import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
llm_client = genai.GenerativeModel('gemini-pro')

# --- FastAPI App Setup ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)

class LessonContent(BaseModel):
    content: str

@app.post("/process_lesson")
def process_lesson(lesson: LessonContent):
    """
    Processes the lesson content to generate a summary and questions using the Gemini API.
    """
    try:
        logging.info("Processing lesson content with Gemini API...")
        
        prompt = f"Summarize the following lesson and generate 3-5 multiple choice questions with answers, formatted as a single JSON object with keys 'summary' and 'questions'. The 'questions' key should be a list of objects, each with 'question', 'options', and 'answer' keys:\n\n{lesson.content}"
        
        response = llm_client.generate_content(prompt)
        
        # Clean up the response to ensure it's valid JSON
        processed_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        logging.info("Successfully processed lesson content with Gemini API.")
        # The response from Gemini should be a JSON string, which we return directly.
        return {"processed_content": processed_text}
        
    except Exception as e:
        logging.error(f"Error processing lesson with Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Failed to process lesson content.")

@app.get("/")
def read_root():
    return {"message": "NLP Content Agent with Gemini is running."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
