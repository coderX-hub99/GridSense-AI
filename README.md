# ⚡ GridSense AI

### Multivariate Electricity Demand Forecasting, Peak Prediction & Anomaly Detection

GridSense AI is a deep learning project that analyzes historical electricity demand and uses **RNN, LSTM, and GRU** models to forecast future electricity demand.

The project also includes a **GRU-based peak demand classifier** and a **forecast-error-based anomaly detection system**.

The trained models are served through a **FastAPI REST API**.

---

## 🚀 Features

* Electricity demand forecasting
* RNN baseline model
* LSTM forecasting model
* GRU forecasting model
* RNN vs LSTM vs GRU comparison
* Peak demand classification
* Forecast-error-based anomaly detection
* Time-series feature engineering
* GPU acceleration with PyTorch CUDA
* FastAPI REST API
* Interactive Swagger API documentation

---

## 🧠 Machine Learning Pipeline

```text
Historical Electricity Data
            ↓
      Data Cleaning
            ↓
    Feature Engineering
            ↓
    Train / Validation / Test
            ↓
       Feature Scaling
            ↓
      24-Hour Sequences
            ↓
     ┌──────┼──────┐
     ↓      ↓      ↓
    RNN    LSTM    GRU
     │      │      │
     └──────┼──────┘
            ↓
      Model Comparison
            ↓
       Best Model: GRU
            │
      ┌─────┴──────────┐
      ↓                ↓
Peak Classification   Anomaly Detection
      │                │
      └───────┬────────┘
              ↓
          FastAPI API
```

---

## 📊 Dataset

The project uses hourly electricity demand data.

### Main columns

* `Datetime`
* `PJM_Load_MW`

The dataset contains approximately **32,900 hourly observations** after restoring missing hourly timestamps.

### Target

```text
PJM_Load_MW
```

The target represents electricity demand measured in megawatts (MW).

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Converted `Datetime` into datetime format.
2. Checked duplicate timestamps.
3. Detected missing hourly timestamps.
4. Restored the complete hourly time index.
5. Filled missing demand values using linear interpolation.
6. Created time-based features.

### Engineered features

```text
hour
day_of_week
month
day
is_weekend
hour_sin
hour_cos
```

The final deep learning input features were:

```text
PJM_Load_MW
hour_sin
hour_cos
day_of_week
month
is_weekend
```

---

## ⏱️ Sequence Creation

The models use the previous **24 hours** of information to predict the next hour.

```text
Previous 24 Hours
       ↓
     RNN/LSTM/GRU
       ↓
Next Hour Demand
```

Input shape:

```text
(batch_size, 24, 6)
```

---

## 🤖 Models

### 1. RNN

A basic recurrent neural network was used as the baseline forecasting model.

### 2. LSTM

LSTM was used to better handle sequential dependencies and long-term patterns.

### 3. GRU

GRU was trained as a simpler recurrent architecture with efficient memory handling.

---

## 📈 Forecasting Results

Test-set performance:

| Model   |   MAE (MW) |  RMSE (MW) |         R² |
| ------- | ---------: | ---------: | ---------: |
| RNN     |     241.64 |     321.77 |     0.9977 |
| LSTM    |     220.18 |     302.53 |     0.9979 |
| **GRU** | **211.50** | **293.14** | **0.9981** |

### 🏆 Best Model

**GRU**

The GRU achieved the lowest MAE and RMSE and the highest R² among the three forecasting models.

---

## 🔥 Peak Demand Prediction

A separate GRU classifier predicts whether the upcoming demand belongs to the **peak-demand category**.

The peak threshold was calculated using the **90th percentile of the training data**.

```text
Normal Demand → 0
Peak Demand   → 1
```

Class weighting was used during training to handle the imbalance between normal and peak observations.

---

## 🚨 Anomaly Detection

GridSense AI uses forecasting error for anomaly detection.

```text
Actual Demand
      ↓
Compare
      ↑
Predicted Demand
      ↓
Absolute Error
      ↓
Anomaly Threshold
      ↓
Normal / Anomaly
```

The anomaly threshold is based on forecasting errors.

A large difference between actual and predicted demand can indicate unusual demand behavior.

---

## ⚡ FastAPI

The trained models are exposed through a REST API using FastAPI.

### Available endpoints

| Method | Endpoint        | Purpose                            |
| ------ | --------------- | ---------------------------------- |
| GET    | `/`             | API information                    |
| GET    | `/health`       | API/model health                   |
| POST   | `/forecast`     | Predict next-hour demand           |
| POST   | `/predict-peak` | Predict peak demand                |
| POST   | `/anomaly`      | Detect anomaly from forecast error |

---

## 📌 API Examples

### Forecast

```text
POST /forecast
```

Requires exactly 24 hours of historical data.

Example response:

```json
{
  "predicted_demand_mw": 48532.21
}
```

---

### Peak Prediction

```text
POST /predict-peak
```

Example response:

```json
{
  "prediction": "Peak Demand",
  "is_peak": true,
  "peak_probability": 0.8734,
  "peak_threshold_mw": 45000.0
}
```

---

### Anomaly Detection

```text
POST /anomaly
```

Example request:

```json
{
  "actual_load_mw": 50000,
  "predicted_load_mw": 48000
}
```

Example response:

```json
{
  "result": "Normal",
  "is_anomaly": false,
  "actual_load_mw": 50000,
  "predicted_load_mw": 48000,
  "forecast_error_mw": 2000,
  "anomaly_threshold_mw": 3500
}
```

The exact threshold and prediction values depend on the trained model and saved artifacts.

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* CUDA
* Matplotlib
* Seaborn
* FastAPI
* Uvicorn
* Jupyter Notebook

---

## 💻 Hardware

Training was performed using:

```text
GPU: NVIDIA GeForce RTX 3050 6GB
RAM: 16GB
```

PyTorch CUDA acceleration was used when available.

---

## 📁 Project Structure

```text
GridSense-AI/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── GridSense_AI.ipynb
│
├── models/
│   ├── gru_best_forecasting.pth
│   ├── peak_gru_classifier.pth
│   ├── feature_scaler.pkl
│   ├── target_scaler.pkl
│   ├── peak_threshold.pkl
│   └── anomaly_threshold.pkl
│
└── screenshots/
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Go into the project:

```bash
cd GridSense-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

## 🚀 Run the API

From the project root:

```powershell
python -m uvicorn app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 📚 Learning Outcomes

Through this project, I worked with:

* Time-series data preprocessing
* Feature engineering
* Data leakage prevention
* Train/validation/test splitting
* Feature scaling
* Sequence generation
* PyTorch tensors and DataLoaders
* RNN
* LSTM
* GRU
* Regression
* Binary classification
* Class imbalance
* Model evaluation
* Anomaly detection
* Model serialization
* FastAPI
* REST API development
* GPU/CUDA acceleration

---

## 🎯 Future Improvements

* Multi-step electricity demand forecasting
* Improved anomaly detection
* Real-time data integration
* Docker containerization
* Cloud deployment
* Monitoring and logging

---

## 👨‍💻 Project

**GridSense AI**

A deep learning project combining time-series forecasting, peak demand classification, anomaly detection, and FastAPI deployment.


