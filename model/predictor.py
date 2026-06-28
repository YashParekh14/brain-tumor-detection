"""
model/predictor.py

Loads the trained EfficientNet model and makes predictions.

This is called by the FastAPI backend (api/main.py)
when a doctor uploads an MRI scan.

Flow:
  MRI image → preprocess → EfficientNet → prediction + confidence
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ─── PATHS ──────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "best_model.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")

# ─── LOAD CONFIG ────────
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

CLASS_NAMES = config["class_names"]   # ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = config["num_classes"]   # 4
IMG_SIZE    = config["img_size"]      # 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── CLASS INFO ───────────
# Extra info shown in the UI for each tumor type
CLASS_INFO = {
    "glioma": {
        "display_name": "Glioma",
        "description":  "A tumor that starts in the glial cells of the brain or spine.",
        "severity":     "High",
        "color":        "#ef4444"
    },
    "meningioma": {
        "display_name": "Meningioma",
        "description":  "A tumor that forms on the membranes surrounding the brain and spinal cord.",
        "severity":     "Medium",
        "color":        "#f97316"
    },
    "notumor": {
        "display_name": "No Tumor",
        "description":  "No tumor detected in the MRI scan.",
        "severity":     "None",
        "color":        "#22c55e"
    },
    "pituitary": {
        "display_name": "Pituitary Tumor",
        "description":  "A tumor that grows in the pituitary gland at the base of the brain.",
        "severity":     "Medium",
        "color":        "#f97316"
    }
}

# ─── IMAGE PREPROCESSING ───────────
# Same transforms as used during testing (NO augmentation)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ─── MODEL LOADER ──────────────
def load_model():
    """
    Load the trained EfficientNet-B0 model from disk.
    Called once when FastAPI server starts.
    """
    # Build same architecture as training
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES)
    )

    # Load saved weights
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()  # Set to evaluation mode
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"✅ Classes: {CLASS_NAMES}")
    print(f"✅ Device:  {DEVICE}")
    return model.to(DEVICE)


# ─── PREDICTION ─────────────────
def predict(image: Image.Image, model) -> dict:
    """
    Takes a PIL image, runs it through EfficientNet,
    returns prediction with confidence scores.

    Args:
        image: PIL Image (the uploaded MRI scan)
        model: Loaded PyTorch model

    Returns:
        {
            "predicted_class": "glioma",
            "display_name":    "Glioma",
            "confidence":      92.15,
            "probabilities": {
                "glioma":      92.15,
                "meningioma":  2.60,
                "notumor":     5.10,
                "pituitary":   0.15
            },
            "info": {
                "description": "A tumor that starts in...",
                "severity":    "High",
                "color":       "#ef4444"
            }
        }
    """
    # Convert grayscale MRI to RGB (model expects 3 channels)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Preprocess: resize → tensor → normalize
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # Run through model
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    # Get top prediction
    predicted_idx   = probs.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = probs[predicted_idx].item() * 100

    # Build probability dict for all classes
    probabilities = {
        CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
        for i in range(NUM_CLASSES)
    }

    return {
        "predicted_class": predicted_class,
        "display_name":    CLASS_INFO[predicted_class]["display_name"],
        "confidence":      round(confidence, 2),
        "probabilities":   probabilities,
        "info":            CLASS_INFO[predicted_class]
    }
