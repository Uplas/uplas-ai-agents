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

class ProjectRequest(BaseModel):
    topic: str
    learning_goal: str

@app.post("/generate_project")
def generate_project(request: ProjectRequest):
    """
    Generates a project plan using the Gemini API.
    """
    try:
        logging.info(f"Generating project for topic: {request.topic} with Gemini API")
        
        prompt = f"Generate a detailed, step-by-step project plan for a beginner learning about '{request.topic}' with the primary goal: '{request.learning_goal}'. The output must be a single, valid JSON object with keys 'project_name', 'description', 'learning_objectives' (a list of strings), and 'project_plan' (a list of objects, where each object has 'step', 'title', and 'description' keys)."
        
        response = llm_client.generate_content(prompt)
        
        # Clean up the response to ensure it's valid JSON
        project_plan_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        logging.info("Successfully generated project plan with Gemini API.")
        # In a real app, you would parse and validate the JSON before returning
        import json
        return json.loads(project_plan_str)

    except Exception as e:
        logging.error(f"Error generating project with Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate project plan.")

@app.get("/")
def read_root():
    return {"message": "Project Generator Agent with Gemini is running."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)
