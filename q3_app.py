import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import gaussian_kde


st.set_page_config(page_title="Interactive Sunspots Analysis", layout="wide")
st.title("Interactive Sunspots Data Dashboard")


@st.cache_data
def load_data(file_path: str = "data/sunspots.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["SUNACTIVITY"] = pd.to_numeric(df["SUNACTIVITY"], errors="coerce")
    df = df.dropna(subset=["YEAR", "SUNACTIVITY"]).copy()
    df["YEAR_INT"] = df["YEAR"].astype(int)
    df["DATE"] = pd.to_datetime(df["YEAR_INT"].astype(str), format="%Y")
    df = df.set_index("DATE").sort_index()
    return df


def plot_advanced_sunspot_visualizations(
    df: pd.DataFrame,
    hist_bins: int = 30,
    trend_degree: int = 1,
    point_size: int = 10,
    point_alpha: float = 0.5,
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Sunspots Data Advanced Visualization", fontsize=18)

    axs[0, 0].plot(df.index, df["SUNACTIVITY"], color="blue")
    axs[0, 0].set_title("Sunspot Activity Over Time")
    axs[0, 0].set_xlabel("Year")
    axs[0, 0].set_ylabel("Sunspot Count")
    axs[0, 0].grid(True)

    data = df["SUNACTIVITY"].dropna().values
    if len(data) > 1:
        xs = np.linspace(data.min(), data.max(), 200)
        density = gaussian_kde(data)
        axs[0, 1].hist(data, bins=hist_bins, density=True, alpha=0.6, color="gray", label="Histogram")
        axs[0, 1].plot(xs, density(xs), color="red", linewidth=2, label="Density")
    axs[0, 1].set_title("Distribution of Sunspot Activity")
    axs[0, 1].set_xlabel("Sunspot Count")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    df_20th = df.loc["1900":"2000"]
    if not df_20th.empty:
        axs[1, 0].boxplot(df_20th["SUNACTIVITY"].dropna(), vert=False)
    axs[1, 0].set_title("Boxplot of Sunspot Activity (1900-2000)")
    axs[1, 0].set_xlabel("Sunspot Count")

    valid_df = df[["YEAR", "SUNACTIVITY"]].dropna()
    years = valid_df["YEAR"].values
    sun_activity = valid_df["SUNACTIVITY"].values
    axs[1, 1].scatter(years, sun_activity, s=point_size, alpha=point_alpha, label="Data Points")
    if len(valid_df) > trend_degree:
        coef = np.polyfit(years, sun_activity, trend_degree)
        trend = np.poly1d(coef)
        x_trend = np.linspace(years.min(), years.max(), 100)
        axs[1, 1].plot(x_trend, trend(x_trend), color="red", linewidth=2, label="Trend Line")
    axs[1, 1].set_title("Trend of Sunspot Activity")
    axs[1, 1].set_xlabel("Year")
    axs[1, 1].set_ylabel("Sunspot Count")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


df = load_data()

st.sidebar.header("Visualization Options")
min_year = int(df["YEAR"].min())
max_year = int(df["YEAR"].max())
year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
hist_bins = st.sidebar.slider("Histogram Bins", 5, 100, 30)
trend_degree = st.sidebar.slider("Trend Degree", 1, 5, 1)
point_size = st.sidebar.slider("Point Size", 5, 100, 10)
point_alpha = st.sidebar.slider("Point Alpha", 0.1, 1.0, 0.5, 0.1)

filtered_df = df[(df["YEAR"] >= year_range[0]) & (df["YEAR"] <= year_range[1])]
st.dataframe(filtered_df.head())
st.pyplot(
    plot_advanced_sunspot_visualizations(
        filtered_df,
        hist_bins=hist_bins,
        trend_degree=trend_degree,
        point_size=point_size,
        point_alpha=point_alpha,
    )
)
