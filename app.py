
# ============================================================
# GridSense AI
# Electricity Demand Forecasting + Peak Prediction + Anomaly Detection
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import torch
import torch.nn as nn
import joblib
import numpy as np


# ============================================================
# 2. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="GridSense AI",
    description="Electricity Demand Forecasting API",
    version="1.0.0"
)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 4. FORECASTING MODEL
# ============================================================

class GRUModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        out, hidden = self.gru(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


# ============================================================
# 5. PEAK CLASSIFICATION MODEL
# ============================================================

class GRUPeakClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        out, hidden = self.gru(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


# ============================================================
# 6. LOAD FORECASTING MODEL
# ============================================================

forecast_model = GRUModel(
    input_size=6,
    hidden_size=64,
    output_size=1
).to(device)


forecast_model.load_state_dict(
    torch.load(
        "models/gru_best_forecasting.pth",
        map_location=device
    )
)

forecast_model.eval()

print("GRU forecasting model loaded successfully.")


# ============================================================
# 7. LOAD PEAK MODEL
# ============================================================

peak_model = GRUPeakClassifier(
    input_size=6,
    hidden_size=64,
    num_classes=2
).to(device)


peak_model.load_state_dict(
    torch.load(
        "models/peak_gru_classifier.pth",
        map_location=device
    )
)

peak_model.eval()

print("GRU peak classifier loaded successfully.")


# ============================================================
# 8. LOAD SCALERS
# ============================================================

feature_scaler = joblib.load(
    "models/feature_scaler.pkl"
)

target_scaler = joblib.load(
    "models/target_scaler.pkl"
)

print("Scalers loaded successfully.")


# ============================================================
# 9. LOAD THRESHOLDS
# ============================================================

peak_threshold = joblib.load(
    "models/peak_threshold.pkl"
)

anomaly_threshold = joblib.load(
    "models/anomaly_threshold.pkl"
)

print("Thresholds loaded successfully.")


# ============================================================
# 10. INPUT DATA MODEL
# ============================================================

class HourData(BaseModel):

    load_mw: float
    hour: int
    day_of_week: int
    month: int
    is_weekend: int


# ============================================================
# 11. FORECAST REQUEST
# ============================================================

class ForecastRequest(BaseModel):

    history: List[HourData]


# ============================================================
# 12. ANOMALY REQUEST
# ============================================================

class AnomalyRequest(BaseModel):

    actual_load_mw: float
    predicted_load_mw: float


# ============================================================
# 13. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "GridSense AI API is running"
    }


# ============================================================
# 14. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "forecast_model": "GRU",
        "peak_model": "GRU",
        "device": str(device)
    }


# ============================================================
# 15. CREATE FEATURES
# ============================================================

def create_features(history):

    feature_rows = []

    for hour_data in history:

        hour_sin = np.sin(
            2 * np.pi * hour_data.hour / 24
        )

        hour_cos = np.cos(
            2 * np.pi * hour_data.hour / 24
        )

        feature_rows.append([
            hour_data.load_mw,
            hour_sin,
            hour_cos,
            hour_data.day_of_week,
            hour_data.month,
            hour_data.is_weekend
        ])

    return np.array(
        feature_rows,
        dtype=np.float32
    )


# ============================================================
# 16. FORECAST API
# ============================================================

@app.post("/forecast")
def forecast(request: ForecastRequest):

    # We need exactly 24 hours
    if len(request.history) != 24:

        return {
            "error": "Exactly 24 hours of history are required."
        }

    # Create six input features
    features = create_features(
        request.history
    )

    # Apply the same scaler used during training
    features_scaled = feature_scaler.transform(
        features
    )

    # Convert NumPy array to PyTorch tensor
    X = torch.tensor(
        features_scaled,
        dtype=torch.float32
    )

    # Add batch dimension
    X = X.unsqueeze(0).to(device)

    # Disable gradient calculation
    with torch.no_grad():

        prediction_scaled = forecast_model(X)

    # Convert prediction back to MW
    prediction_mw = target_scaler.inverse_transform(
        prediction_scaled.cpu().numpy()
    )

    predicted_load = float(
        prediction_mw[0][0]
    )

    return {
        "predicted_demand_mw": round(
            predicted_load,
            2
        )
    }


# ============================================================
# 17. PEAK PREDICTION API
# ============================================================

@app.post("/predict-peak")
def predict_peak(request: ForecastRequest):

    # Peak model also needs 24 hours
    if len(request.history) != 24:

        return {
            "error": "Exactly 24 hours of history are required."
        }

    # Create six features
    features = create_features(
        request.history
    )

    # Scale features
    features_scaled = feature_scaler.transform(
        features
    )

    # Convert to tensor
    X = torch.tensor(
        features_scaled,
        dtype=torch.float32
    )

    # Add batch dimension
    X = X.unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():

        logits = peak_model(X)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    # Probability of peak class
    peak_probability = float(
        probabilities[0][1].item()
    )

    # Convert class into readable result
    if predicted_class == 1:

        result = "Peak Demand"

    else:

        result = "Normal Demand"

    return {
        "prediction": result,
        "is_peak": bool(predicted_class),
        "peak_probability": round(
            peak_probability,
            4
        ),
        "peak_threshold_mw": round(
            float(peak_threshold),
            2
        )
    }


# ============================================================
# 18. ANOMALY DETECTION API
# ============================================================

@app.post("/anomaly")
def detect_anomaly(request: AnomalyRequest):

    # Calculate absolute forecasting error
    error = abs(
        request.actual_load_mw
        - request.predicted_load_mw
    )

    # Compare error with threshold
    is_anomaly = (
        error >= anomaly_threshold
    )

    # Readable result
    if is_anomaly:

        result = "Anomaly Detected"

    else:

        result = "Normal"

    return {
        "result": result,
        "is_anomaly": bool(is_anomaly),
        "actual_load_mw": request.actual_load_mw,
        "predicted_load_mw": request.predicted_load_mw,
        "forecast_error_mw": round(
            error,
            2
        ),
        "anomaly_threshold_mw": round(
            float(anomaly_threshold),
            2
        )
    }
