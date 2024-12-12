from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
from utils.inference import (
    predict_baseline,
    predict_content_based,
    predict_collaborative,
    predict_session_based
)

app = FastAPI()

# Входные данные
class FeaturesInput(BaseModel):
    features: List[List[float]]
    model_type: Optional[str] = "baseline"  # Тип модели: baseline, content, collaborative, session

@app.post("/predict")
def predict(input_data: FeaturesInput):
    try:
        features = np.array(input_data.features, dtype=np.float32)
        model_type = input_data.model_type.lower()

        print("Features:", features)
        print("Model type:", model_type)

        # Выбор модели
        if model_type == "baseline":
            predictions = predict_baseline(features)
        elif model_type == "content":
            predictions = predict_content_based(features)
        elif model_type == "collaborative":
            predictions = predict_collaborative(features)
        elif model_type == "session":
            predictions = predict_session_based(features)
        else:
            return {"status": "error", "message": f"Unknown model type: {model_type}"}

        return {"status": "success", "model_type": model_type, "predictions": predictions.tolist()}
    except Exception as e:
        return {"status": "error", "message": str(e)}