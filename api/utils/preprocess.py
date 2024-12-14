from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Optional
from utils.inference import (
    predict_baseline,
    predict_content_based,
    predict_collaborative,
    predict_session_based
)

app = FastAPI()

class PredictionInput(BaseModel):
    anime_id: int = Field(..., description="Anime ID to get recommendations for")
    top_n: int = Field(default=5, ge=1, le=100, description="Number of recommendations to return")
    model_type: Optional[str] = Field(
        default="baseline",
        description="Model type to use for predictions"
    )

    class Config:
        schema_extra = {
            "example": {
                "anime_id": 1234,
                "top_n": 5,
                "model_type": "baseline"
            }
        }

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        if input_data.top_n < 1 or input_data.top_n > 100:
            raise HTTPException(
                status_code=400,
                detail="top_n must be between 1 and 100"
            )

        # Simple 2D array, preprocessing moved to predict_baseline
        anime_id = np.array([[input_data.anime_id]], dtype=np.float32)
        model_type = input_data.model_type.lower()

        print(f"Request - Anime ID: {input_data.anime_id}, Top N: {input_data.top_n}")
        print("Model type:", model_type)

        if model_type == "baseline":
            predictions = predict_baseline(anime_id, input_data.top_n)
        elif model_type == "content":
            predictions = predict_content_based(anime_id)
        elif model_type == "collaborative":
            predictions = predict_collaborative(anime_id)
        elif model_type == "session":
            predictions = predict_session_based(anime_id)
        else:
            return {"status": "error", "message": f"Unknown model type: {model_type}"}

        return {
            "status": "success",
            "model_type": model_type,
            "predictions": predictions.tolist()[:input_data.top_n]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )