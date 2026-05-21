import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet


st.set_page_config(page_title="Sunspot Forecast", layout="wide")
st.title("Prophet Forecast with Sunspot Data")


@st.cache_data
def load_prophet_data() -> pd.DataFrame:
    try:
        df = pd.read_csv("data/sunspots_for_prophet.csv")
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    except FileNotFoundError:
        raw = pd.read_csv("data/sunspots.csv")
        raw.columns = [col.strip() for col in raw.columns]
        raw["YEAR"] = pd.to_numeric(raw["YEAR"], errors="coerce")
        raw["SUNACTIVITY"] = pd.to_numeric(raw["SUNACTIVITY"], errors="coerce")
        raw = raw.dropna(subset=["YEAR", "SUNACTIVITY"]).copy()
        raw["ds"] = pd.to_datetime(raw["YEAR"].astype(int).astype(str), format="%Y")
        df = raw.rename(columns={"SUNACTIVITY": "y"})[["ds", "y"]]
        df = df[(df["ds"] >= "1900-01-01") & (df["ds"] <= "2008-01-01")]
        return df.reset_index(drop=True)


@st.cache_resource
def train_model(df: pd.DataFrame) -> Prophet:
    model = Prophet(
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode="additive",
    )
    model.add_seasonality(name="sunspot_cycle", period=365.25 * 11, fourier_order=5)
    model.fit(df)
    return model


df = load_prophet_data()
st.subheader("Data Preview")
st.dataframe(df.head())

model = train_model(df)
future = model.make_future_dataframe(periods=30, freq="Y")
forecast = model.predict(future)

st.subheader("Prophet Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

st.subheader("Actual vs Predicted with Prediction Intervals")
fig3, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["ds"], df["y"], label="Actual", color="tab:blue")
ax.plot(forecast["ds"], forecast["yhat"], label="Predicted", color="tab:orange")
ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    color="tab:orange",
    alpha=0.2,
    label="Prediction Interval",
)
ax.set_xlabel("Year")
ax.set_ylabel("Sunspot Activity")
ax.legend()
ax.grid(True)
st.pyplot(fig3)

st.subheader("Residual Analysis")
merged = pd.merge(df, forecast[["ds", "yhat"]], on="ds", how="inner")
merged["residual"] = merged["y"] - merged["yhat"]

fig4, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(merged["ds"], merged["residual"], label="Residual", color="tab:purple")
ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
ax2.set_xlabel("Year")
ax2.set_ylabel("Residual")
ax2.legend()
ax2.grid(True)
st.pyplot(fig4)

st.subheader("Residual Summary Statistics")
st.write(merged["residual"].describe())
