"""
src/rag/tools.py — Custom LangChain tools wrapper for the pipeline.
"""

import os
from langchain_core.tools import tool
from src.rag.model.vision import SkinVisionPredictor

# Dynamic path taake local aur production server dono par model mil jaye
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_FILE_PATH = os.path.join(ROOT_DIR, "best_skin_issues_model_v2.pth")

predictor = SkinVisionPredictor(model_path=MODEL_FILE_PATH)


@tool
def analyze_skin_image(image_path: str) -> str:
    """
    Analyzes an uploaded skin image (local path or Supabase URL) using an EfficientNet-B0 model.
    Returns the predicted clinical skin condition category.
    """
    is_url = isinstance(image_path, str) and image_path.startswith(("http://", "https://"))
    if not is_url and not os.path.exists(image_path):
        return f"Error: Image file at path {image_path} could not be located."

    try:
        prediction_result = predictor.predict_skin_issue(image_path)
        return prediction_result["class"]
    except Exception as e:
        return f"Error executing vision inference: {str(e)}"
