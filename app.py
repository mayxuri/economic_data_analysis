# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import json

st.set_page_config(page_title="🌍 Cost of Living Dashboard", layout="wide")

# -------------------------------
# Load Data + Model
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Cost_of_Living_Index_2022.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/cost_of_living_model.pkl")

@st.cache_data
def load_metrics():
    try:
        with open("models/training_metrics.json", "r") as f:
            return json.load(f)
    except:
        return {}

df = load_data()
model_dict = load_model()
metrics = load_metrics()

model = model_dict["model"]
features = model_dict["features"]
target = model_dict["target"]

# -------------------------------
# Sidebar Navigation
# -------------------------------
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Dataset Viewer",
    "Country Explorer",
    "Prediction",
    "Training Results"
])

# -------------------------------
# Overview Page
# -------------------------------
if page == "Overview":
    st.title("🌍 Cost of Living Analysis")
    st.write("Explore the 2022 Cost of Living dataset and ML predictions.")

    st.subheader("Top 10 Countries by Cost of Living")
    top10 = df.nlargest(10, target)
    fig = px.bar(top10, x="Country", y=target, color=target, text=target)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes("number").corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Dataset Viewer Page
# -------------------------------
elif page == "Dataset Viewer":
    st.title("📊 Dataset Viewer")
    st.write("Browse the dataset below:")
    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "cost_of_living_data.csv",
        "text/csv"
    )

# -------------------------------
# Country Explorer Page
# -------------------------------
elif page == "Country Explorer":
    st.title("🔎 Country Explorer")
    country = st.selectbox("Select a country", df["Country"].unique())
    row = df[df["Country"] == country].T.reset_index()
    row.columns = ["Metric", "Value"]
    st.table(row)

    fig = px.bar(row[1:], x="Metric", y="Value", title=f"{country} - Indices")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Prediction Page
# -------------------------------
elif page == "Prediction":
    st.title("🤖 Predict Cost of Living Index")
    st.write("Adjust the sliders to predict the index.")

    inputs = {}
    for feat in features:
        inputs[feat] = st.slider(
            feat,
            float(df[feat].min()), float(df[feat].max()), float(df[feat].mean())
        )

    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]

    st.metric("Predicted Cost of Living Index", f"{prediction:.2f}")
    st.write("Inputs used for prediction:")
    st.dataframe(input_df)

# -------------------------------
# Training Results Page
# -------------------------------
elif page == "Training Results":
    st.title("📈 Model Training Results")

    if metrics:
        st.json(metrics)
    else:
        st.warning("No training metrics found. Run train_model.py to generate metrics.")
