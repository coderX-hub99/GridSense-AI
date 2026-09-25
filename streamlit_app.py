"""Browser interface for the GridSense AI models.

Run with:
    streamlit run streamlit_app.py
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import (  # noqa: E402
    AnomalyRequest,
    ForecastRequest,
    HourData,
    detect_anomaly,
    forecast,
    predict_peak,
)


st.set_page_config(
    page_title="GridSense AI",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ GridSense AI")
st.caption("Electricity demand forecasting, peak prediction, and anomaly detection")


@st.cache_data
def load_default_data() -> pd.DataFrame:
    data_path = PROJECT_DIR / "PJM_Load_hourly.csv"
    data = pd.read_csv(data_path)
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
    data["PJM_Load_MW"] = pd.to_numeric(data["PJM_Load_MW"], errors="coerce")
    return data.dropna(subset=["Datetime", "PJM_Load_MW"]).sort_values("Datetime")


def prepare_history(data: pd.DataFrame, end_row: int) -> list[HourData]:
    window = data.iloc[end_row - 24:end_row]
    history = []
    for row in window.itertuples(index=False):
        timestamp = row.Datetime
        history.append(
            HourData(
                load_mw=float(row.PJM_Load_MW),
                hour=int(timestamp.hour),
                day_of_week=int(timestamp.dayofweek),
                month=int(timestamp.month),
                is_weekend=int(timestamp.dayofweek >= 5),
            )
        )
    return history


with st.sidebar:
    st.header("Input data")
    uploaded_file = st.file_uploader("Upload hourly CSV (optional)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if "Datetime" not in data.columns or "PJM_Load_MW" not in data.columns:
            st.error("CSV must contain Datetime and PJM_Load_MW columns.")
            st.stop()
        data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
        data["PJM_Load_MW"] = pd.to_numeric(data["PJM_Load_MW"], errors="coerce")
        data = data.dropna(subset=["Datetime", "PJM_Load_MW"]).sort_values("Datetime")
    else:
        data = load_default_data()

if len(data) < 24:
    st.error("At least 24 valid hourly observations are required.")
    st.stop()

st.subheader("Recent electricity demand")
st.line_chart(data.set_index("Datetime")["PJM_Load_MW"].tail(168))

max_end_row = len(data)
default_end_row = max(24, max_end_row - 1)
end_row = st.slider(
    "Use the 24 hours before this row as model input",
    min_value=24,
    max_value=max_end_row,
    value=default_end_row,
)
history = prepare_history(data, end_row)
next_timestamp = data.iloc[end_row]["Datetime"] if end_row < len(data) else None

if next_timestamp is not None:
    st.info(f"Forecasting the hour after {data.iloc[end_row - 1]['Datetime']}.")
else:
    st.info("Forecasting the next hour after the latest available observation.")

if st.button("Run forecast and peak prediction", type="primary"):
    forecast_result = forecast(ForecastRequest(history=history))
    peak_result = predict_peak(ForecastRequest(history=history))
    if "error" in forecast_result or "error" in peak_result:
        st.error(forecast_result.get("error") or peak_result.get("error"))
    else:
        st.session_state["forecast_result"] = forecast_result
        st.session_state["peak_result"] = peak_result

forecast_result = st.session_state.get("forecast_result")
peak_result = st.session_state.get("peak_result")
if forecast_result and peak_result:
    predicted_load = forecast_result["predicted_demand_mw"]
    metric_col, peak_col = st.columns(2)
    metric_col.metric("Next-hour demand", f"{predicted_load:,.2f} MW")
    peak_label = "Peak demand" if peak_result["is_peak"] else "Normal demand"
    peak_col.metric("Peak classification", peak_label)
    st.progress(
        peak_result["peak_probability"],
        text=f"Peak probability: {peak_result['peak_probability']:.1%}",
    )
    st.caption(f"Peak threshold: {peak_result['peak_threshold_mw']:,.2f} MW")

    st.subheader("Anomaly detection")
    actual_load = st.number_input(
        "Actual demand for the forecasted hour (MW)",
        min_value=0.0,
        value=float(predicted_load),
        step=100.0,
    )
    if st.button("Check anomaly"):
        anomaly_result = detect_anomaly(
            AnomalyRequest(
                actual_load_mw=actual_load,
                predicted_load_mw=predicted_load,
            )
        )
        if anomaly_result["is_anomaly"]:
            st.error(
                f"Anomaly detected: forecast error "
                f"{anomaly_result['forecast_error_mw']:,.2f} MW."
            )
        else:
            st.success(
                f"Normal: forecast error "
                f"{anomaly_result['forecast_error_mw']:,.2f} MW."
            )
        st.caption(
            f"Anomaly threshold: {anomaly_result['anomaly_threshold_mw']:,.2f} MW"
        )

    fig, ax = plt.subplots(figsize=(10, 3))
    recent = data.iloc[max(0, end_row - 48):end_row]
    ax.plot(recent["Datetime"], recent["PJM_Load_MW"], label="Historical demand")
    forecast_time = recent["Datetime"].iloc[-1] + pd.Timedelta(hours=1)
    ax.scatter([forecast_time], [predicted_load], label="Forecast", color="tab:red")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
