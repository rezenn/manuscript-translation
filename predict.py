"""
predict.py  —  Newa Script OCR
Tests your trained model on a single image.

Usage:
    python predict.py path/to/your/character_image.png

This is useful for:
  - Checking if your model works before building the full pipeline
  - Debugging individual misclassifications
  - Seeing confidence scores for each class
"""

import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import build_model

# ── Settings ──────────────────────────────────────────────────────────────────

MODEL_PATH    = "model_best.pth"
CLASS_FILE    = "class_names.txt"
IMAGE_SIZE    = 128
TOP_K         = 5          # show top 5 predictions with their confidence %

# ── Load class names ──────────────────────────────────────────────────────────

def load_class_names(path=CLASS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(image_path):
    """Loads an image and converts it to the format the model expects."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    image = Image.open(image_path).convert("L")   # convert to grayscale
    return transform(image).unsqueeze(0)           # add batch dimension: [1,1,128,128]

# ── Predict ───────────────────────────────────────────────────────────────────

def predict(image_path, model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names
    class_names = load_class_names()

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Preprocess image
    image_tensor = preprocess_image(image_path).to(device)

    # Run prediction
    with torch.no_grad():
        output = model(image_tensor)                        # raw scores
        probabilities = F.softmax(output, dim=1)[0]        # convert to probabilities

    # Get top K predictions
    top_probs, top_indices = probabilities.topk(TOP_K)

    print(f"\nPrediction for: {image_path}")
    print("-" * 40)
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        class_name = class_names[idx.item()]
        confidence = prob.item() * 100
        marker = "  ← TOP PREDICTION" if rank == 1 else ""
        print(f"  {rank}. {class_name:<30} {confidence:6.2f}%{marker}")

    top_class = class_names[top_indices[0].item()]
    top_conf  = top_probs[0].item() * 100
    return top_class, top_conf


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.png")
        print("\nExample: python predict.py dataset_final/test/class_0/img001.png")
        sys.exit(1)

    image_path = sys.argv[1]
    top_class, top_conf = predict(image_path)
    print(f"\nFinal answer: {top_class} ({top_conf:.1f}% confidence)")