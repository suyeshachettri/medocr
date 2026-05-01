# MedOCR — Medical Handwriting Recognition

AI model that reads messy doctor handwriting using fine-tuned TrOCR + FastAPI.

## Results
| Metric | Score |
|---|---|
| Exact Match Accuracy | 86% |
| Character Error Rate | 6.65% |
| Word Error Rate | 15.46% |

## Tech Stack
- **Model:** Microsoft TrOCR (fine-tuned)
- **Dataset:** IAM Handwriting (34,000+ samples) + Synthetic medical data
- **API:** FastAPI
- **Training:** PyTorch on Google Colab T4 GPU

## Project Structure
```
medocr/
├── app.py                       # FastAPI application
├── test_api.py                  # API test script
├── notebooks/
│   ├── 01_data_setup.ipynb      # Data preparation
│   └── 02_train_trocr.ipynb     # Model training
├── models/trocr-medical/        # Trained model config
└── data/
    └── labels_final.csv         # Dataset labels
```

## Setup
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## API Usage
```bash
# Predict handwriting from image
POST http://127.0.0.1:8000/predict

# Interactive Swagger UI
GET http://127.0.0.1:8000/docs
```

## Sample Predictions
The model correctly reads medical prescriptions like:
- `Tab. Metformin 500mg BD x 30d`
- `Inj. Ceftriaxone 1g IV BD`
- `Syp. Amoxicillin 250mg/5ml TID`