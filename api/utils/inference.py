import onnxruntime
import numpy as np
import os

# Debug: Verify the presence of the model files
print("Model files in 'models' directory:", os.listdir("models"))

# Подгрузка моделей
baseline_session = onnxruntime.InferenceSession("models/baseline_model.onnx")
content_session = onnxruntime.InferenceSession("models/baseline_model.onnx")
collaborative_session = onnxruntime.InferenceSession("models/baseline_model.onnx")
session_based_session = onnxruntime.InferenceSession("models/baseline_model.onnx")

def predict_baseline(features):
    input_name = baseline_session.get_inputs()[0].name
    predictions = baseline_session.run(None, {input_name: features})
    return predictions[0]

def predict_content_based(features):
    input_name = content_session.get_inputs()[0].name
    predictions = content_session.run(None, {input_name: features})
    return predictions[0]

def predict_collaborative(features):
    input_name = collaborative_session.get_inputs()[0].name
    predictions = collaborative_session.run(None, {input_name: features})
    return predictions[0]

def predict_session_based(features):
    input_name = session_based_session.get_inputs()[0].name
    predictions = session_based_session.run(None, {input_name: features})
    return predictions[0]