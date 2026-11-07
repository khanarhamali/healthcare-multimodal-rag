"""
src/rag/tools.py — Custom LangChain tools wrapper for the pipeline.
"""

import os
from langchain_core.tools import tool
from src.rag.model.vision import SkinVisionPredictor

# Initialize your model instance safely
MODEL_FILE_PATH = "E:\\rag-eval\\best_skin_issues_model_v2.pth"
predictor = SkinVisionPredictor(model_path=MODEL_FILE_PATH)

@tool
def analyze_skin_image(image_path: str) -> str:
    """
    Analyzes an uploaded skin image file using an EfficientNet-B0 deep learning model.
    Pass this tool the absolute path to the local image file. 
    It returns the predicted clinical skin condition category.
    """
    if not os.path.exists(image_path):
        return f"Error: Image file at path {image_path} could not be located."
        
    try:
        prediction_result = predictor.predict_skin_issue(image_path)
        return prediction_result["class"]
    except Exception as e:
        return f"Error executing vision inference: {str(e)}"
