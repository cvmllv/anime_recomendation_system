import onnxruntime
import numpy as np
import os
from pathlib import Path

ANIME_FEATURE_SIZE = 13357  # Add this constant at the top with other imports

def load_model(model_path):
    try:
        absolute_path = Path(__file__).parent.parent / model_path
        return onnxruntime.InferenceSession(str(absolute_path))
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None

# Debug: Verify the presence of the model files
models_dir = Path(__file__).parent.parent / "models"
print("Model files in 'models' directory:", list(models_dir.glob("*.onnx")))

# Load models with error handling
baseline_session = load_model("models/svd_model_baseline.onnx")
content_session = load_model("models/baseline_model.onnx")
collaborative_session = load_model("models/tfidf_content_based.onnx")
session_based_session = load_model("models/baseline_model.onnx")

def predict_baseline(anime_id, top_n):
    if baseline_session is None:
        raise RuntimeError("Baseline model failed to load")
    
    # Create zero vector with correct shape
    input_vector = np.zeros((1, ANIME_FEATURE_SIZE), dtype=np.float32)
    
    # Set the anime_id index to 1
    anime_idx = int(anime_id[0][0]) % ANIME_FEATURE_SIZE  # Ensure index is within bounds
    input_vector[0, anime_idx] = 1.0
    
    print(f"Model input shape: {input_vector.shape}, dtype: {input_vector.dtype}")
    print(f"Non-zero indices: {np.nonzero(input_vector)[1]}")
    
    input_name = baseline_session.get_inputs()[0].name
    predictions = baseline_session.run(None, {input_name: input_vector})
    return predictions[0]

def predict_content_based(features):
    if content_session is None:
        raise RuntimeError("Content-based model failed to load")
    input_name = content_session.get_inputs()[0].name
    predictions = content_session.run(None, {input_name: features})
    return predictions[0]

def predict_collaborative(features):
    if collaborative_session is None:
        raise RuntimeError("Collaborative model failed to load")
    input_name = collaborative_session.get_inputs()[0].name
    predictions = collaborative_session.run(None, {input_name: features})
    return predictions[0]

def predict_session_based(features):
    if session_based_session is None:
        raise RuntimeError("Session-based model failed to load")
    input_name = session_based_session.get_inputs()[0].name
    predictions = session_based_session.run(None, {input_name: features})
    return predictions[0]