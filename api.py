# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple # Use Tuple from typing

from utils import mask_pii
from models import load_classifier, predict_category

# Load the classification pipeline on startup
pipeline = load_classifier()

# Initialize FastAPI app
app = FastAPI(
    title="Email Classification and PII Masking API",
    description="Accepts email text, masks PII (non-LLM), classifies the email, and returns results.",
    version="1.0.0"
)

# --- Pydantic Models for Request and Response ---

class EmailRequest(BaseModel):
    email_body: str = Field(..., example="Hello John Doe, please check my invoice #12345 regarding the billing issue. My email is john.d@example.com")

class MaskedEntity(BaseModel):
    position: Tuple[int, int] = Field(..., example=[6, 14]) # Use Tuple[int, int]
    classification: str = Field(..., example="full_name")
    entity: str = Field(..., example="John Doe")

class ClassificationResponse(BaseModel):
    input_email_body: str = Field(..., example="Hello John Doe, please check my invoice #12345...")
    list_of_masked_entities: List[MaskedEntity] = Field(...)
    masked_email: str = Field(..., example="Hello [full_name], please check my invoice #12345...")
    category_of_the_email: str = Field(..., example="Billing Issue")

# --- API Endpoint ---

@app.post("/classify/", response_model=ClassificationResponse)
async def classify_email(request: EmailRequest):
    """
    Receives an email, masks PII, classifies it, and returns structured output.
    """
    if not request.email_body:
        raise HTTPException(status_code=400, detail="email_body cannot be empty")

    try:
        # 1. Mask PII
        masked_text, entities_list = mask_pii(request.email_body)

        # Ensure entities_list format matches Pydantic model (it should from utils.py)
        # Pydantic will validate this structure automatically on return

        # 2. Classify the masked text
        predicted_category = predict_category(masked_text, pipeline)

        # 3. Construct the response
        response_data = ClassificationResponse(
            input_email_body=request.email_body,
            list_of_masked_entities=entities_list, # Pass the list directly
            masked_email=masked_text,
            category_of_the_email=predicted_category
        )

        return response_data

    except Exception as e:
        # Log the exception for debugging
        print(f"Error processing request: {e}")
        # Re-raise as HTTPException for FastAPI to handle
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# --- Root endpoint for basic check ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Email Classification API. Use the /classify/ endpoint."}