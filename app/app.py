import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "game_lifecycle_analyzed.csv"
MODEL_6M_PATH = ROOT / "models" / "lifecycle_6m.pkl"
MODEL_12M_PATH = ROOT / "models" / "lifecycle_12m.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    return joblib.load(MODEL_6M_PATH), joblib.load(MODEL_12M_PATH)

games = load_data()
model_6m, model_12m = load_models()
games["log_baseline_players"] = np.log1p(games["baseline_players"])

st.set_page_config(page_title="Game Lifecycle Predictor", page_icon="🎮", layout="wide")
st.title("🎮 Video Game Lifecycle Predictor")
st.write("Predict whether a game's long-term player lifecycle will be **Declining, Sustained, or Growing** using its early player activity.")

game_title = st.selectbox("Search for a game", sorted(games["title"].dropna().unique()), index=None, placeholder="Type a game title...")

if game_title:
    game = games[games["title"] == game_title].iloc[0]
    st.header(game_title)

    col1, col2, col3 = st.columns(3)
    col1.metric("Launch Price", f"${game['launch_price']:.2f}")
    col2.metric("Baseline Players", f"{game['baseline_players']:,.0f}")
    col3.metric("Multiplayer", "Yes" if game["is_multiplayer"] == 1 else "No")

    st.divider()

    prediction_point = st.radio("Prediction Point", ["6 Months", "12 Months"], horizontal=True)

    if prediction_point == "6 Months":
        features = ["launch_price", "log_baseline_players", "is_multiplayer", "relative_6m"]
        model = model_6m
    else:
        features = ["launch_price", "log_baseline_players", "is_multiplayer", "relative_6m", "relative_12m"]
        model = model_12m

    X_game = pd.DataFrame([game[features].values], columns=features)
    prediction = model.predict(X_game)[0]
    probabilities = model.predict_proba(X_game)[0]
    probability_df = pd.DataFrame({"Lifecycle": model.classes_, "Probability": probabilities})

    st.subheader("Predicted Lifecycle")
    st.markdown(f"# {prediction}")
    st.metric("Model Confidence", f"{probabilities.max():.1%}")

    st.subheader("Prediction Probabilities")
    st.bar_chart(probability_df.set_index("Lifecycle"))

    st.subheader("Early Player Retention")

    if prediction_point == "6 Months":
        st.metric("6-Month Retention", f"{game['relative_6m']:.1%}")
    else:
        col1, col2 = st.columns(2)
        col1.metric("6-Month Retention", f"{game['relative_6m']:.1%}")
        col2.metric("12-Month Retention", f"{game['relative_12m']:.1%}")