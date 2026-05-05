import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, io

app = FastAPI(
    title="MedOCR API",
    description="Medical handwriting recognition API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loads locally in VS Code, downloads from HuggingFace on server
MODEL_DIR = os.environ.get("MODEL_DIR", "suyeshachettri/medocr-trocr")

print(f"Loading model from: {MODEL_DIR}")
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
model.eval()
print("✅ Model ready!")

def predict(image: Image.Image) -> str:
    pixel_values = processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

@app.get("/")
def root():
    return {"message": "MedOCR API is running!", "status": "healthy"}

@app.post("/predict")
async def predict_handwriting(file: UploadFile = File(...)):
    contents       = await file.read()
    image          = Image.open(io.BytesIO(contents)).convert("RGB")
    predicted_text = predict(image)
    return {
        "predicted_text": predicted_text,
        "filename":       file.filename,
        "status":         "success"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "trocr-medical"}