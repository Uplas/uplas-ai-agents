import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Mock LLM Client ---
# This mock client simulates the behavior of the Vertex AI LLM client
# for local development without making any actual API calls.

class MockLLMClient:
    """A mock LLM client that returns predefined responses."""

    def predict(self, prompt: str) -> str:
        """Simulates a prediction by returning a mock response."""
        logging.info(f"MockLLMClient received prompt: {prompt[:100]}...")
        # This is a sample response. You can customize it for your testing needs.
        mock_response = {
            "summary": "This is a mock summary of the lesson.",
            "questions": [
                {
                    "question": "What is the capital of France?",
                    "options": ["Paris", "London", "Berlin", "Madrid"],
                    "answer": "Paris"
                },
                {
                    "question": "What is 2 + 2?",
                    "options": ["3", "4", "5", "6"],
                    "answer": "4"
                }
            ]
        }
        import json
        return json.dumps(mock_response)

# --- FastAPI App Setup ---

app = FastAPI()

# Initialize the mock client
llm_client = MockLLMClient()

logging.basicConfig(level=logging.INFO)

class LessonContent(BaseModel):
    content: str

@app.post("/process_lesson")
def process_lesson(lesson: LessonContent):
    """
    Processes the lesson content to generate a summary and questions.
    This endpoint now uses a mock LLM client for local development.
    """
    try:
        logging.info("Processing lesson content...")
        # The prompt is created, but the mock client will return a fixed response.
        prompt = f"Summarize the following lesson and generate 3-5 multiple choice questions with answers:\n\n{lesson.content}"
        
        response_text = llm_client.predict(prompt)
        
        logging.info("Successfully processed lesson content with mock client.")
        return {"processed_content": response_text}
    except Exception as e:
        logging.error(f"Error processing lesson: {e}")
        raise HTTPException(status_code=500, detail="Failed to process lesson content.")

@app.get("/")
def read_root():
    return {"message": "NLP Content Agent is running in local mode."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
