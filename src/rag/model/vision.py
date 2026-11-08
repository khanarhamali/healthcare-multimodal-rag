import os
import sys
import torch
import requests
from io import BytesIO
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Setup global path checking to keep imports functional
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

class SkinVisionPredictor:
    def __init__(self, model_path="best_skin_issues_model_v2.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["Acne", "Blackheads Whiteheads", "Cyst", 
                            "Darkspots", "Papules Pustules", "Pores", "Wrinkles"]
        
        # 1. Transform setup
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 2. Load PyTorch Model Architecture
        print("Loading EfficientNet-B0 Model Architecture...")
        self.model = self._init_and_load_model(model_path)

    def _init_and_load_model(self, model_path):
        model = models.efficientnet_b0(weights=None)
        
        # Access index 1 of the default classifier to extract the correct in_features
        in_features = model.classifier[1].in_features
        
        # Rebuild model classifier structure
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(self.class_names))
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please verify the absolute path!")

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict_skin_issue(self, image_path):
        """Processes the image from either a web URL (Supabase) or local file path"""
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Test image '{image_path}' not found!")
            image = Image.open(image_path).convert("RGB")

        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return {
            "class": self.class_names[predicted.item()],
            "confidence": float(confidence.item())
        }


if __name__ == "__main__":
    # Test block config
    MODEL_FILE_PATH = "E:\\rag-eval\\best_skin_issues_model_v2.pth"
    TEST_IMAGE_PATH = "E:\\rag-eval\\cystic acne.jpg" 

    # Run Prediction Only
    print(f"Testing Vision Model Prediction on: {os.path.basename(TEST_IMAGE_PATH)}")
    predictor = SkinVisionPredictor(model_path=MODEL_FILE_PATH)
    result = predictor.predict_skin_issue(TEST_IMAGE_PATH)
    
    print("\n================ VISION OUTPUT ================")
    print(f"Detected Condition : {result['class']}")
    print(f"Confidence Score    : {result['confidence']:.2%}")
    print("===============================================")
