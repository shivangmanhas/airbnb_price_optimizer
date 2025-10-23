import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path
import math

st.set_page_config(page_title="Airbnb Price Optimizer", layout="wide")

st.title("🏠 Airbnb Price Optimizer")

st.markdown("""
Upload or point the app to a trained model file (`models/price_model_<city>.joblib`).  
Use the sidebar to enter listing details and get a recommended nightly price.
""")

with st.sidebar:
    st.header("Model")
    city = st.text_input("City code (e.g., nyc)", value="nyc")
    model_path = st.text_input("Path to model", value=f"models/price_model_{city}.joblib")
    st.caption("Train a model with `python -m src.models.train_price_model --city <city>`")

    # Inputs
    st.header("Listing Inputs")
    latitude = st.number_input("Latitude", value=40.7128, format="%.6f")
    longitude = st.number_input("Longitude", value=-74.0060, format="%.6f")
    neighbourhood = st.text_input("Neighbourhood (cleansed)", value="Manhattan")
    accommodates = st.number_input("Accommodates", min_value=1, value=2)
    bedrooms = st.number_input("Bedrooms", min_value=0.0, value=1.0, step=0.5)
    beds = st.number_input("Beds", min_value=0.0, value=1.0, step=0.5)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, value=1.0, step=0.5)
    review_scores_rating = st.number_input("Review Score (0–100)", min_value=0.0, max_value=100.0, value=90.0, step=0.5)
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
    host_listings_count = st.number_input("Host Listings Count", min_value=0, value=1)
    amenities_count = st.number_input("Amenities Count", min_value=0, value=10)
    host_tenure_days = st.number_input("Host Tenure (days)", min_value=0, value=365)
    has_picture = st.selectbox("Has Picture", options=[0,1], index=1)
    room_type = st.selectbox("Room Type", options=["Entire home/apt","Private room","Shared room","Hotel room"], index=0)
    occupancy_rate_mean = st.slider("Avg Occupancy (0-1)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    st.header("Pricing")
    candidate_min = st.number_input("Min candidate price", min_value=10, value=40)
    candidate_max = st.number_input("Max candidate price", min_value=20, value=300)
    candidate_step = st.number_input("Price step", min_value=1, value=5)

# Build feature row compatible with training script
def build_feature_row():
    row = {
        "latitude": latitude,
        "longitude": longitude,
        "neighbourhood_cleansed": neighbourhood,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": beds,
        "bathrooms": bathrooms,
        "review_scores_rating": review_scores_rating,
        "number_of_reviews": number_of_reviews,
        "host_listings_count": host_listings_count,
        "amenities_count": amenities_count,
        "host_tenure_days": host_tenure_days,
        "has_picture": has_picture,
        "occupancy_rate_mean": occupancy_rate_mean,
        # room_type dummies:
        "room_type_Entire home/apt": 1 if room_type=="Entire home/apt" else 0,
        "room_type_Private room": 1 if room_type=="Private room" else 0,
        "room_type_Shared room": 1 if room_type=="Shared room" else 0,
        "room_type_Hotel room": 1 if room_type=="Hotel room" else 0,
    }
    return pd.DataFrame([row])

# Load model if present
model = None
model_fp = Path(model_path)
if model_fp.exists():
    try:
        model = load(model_fp)
        st.success(f"Loaded model: {model_fp}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.warning("Model file not found. Train a model first.")

feat_df = build_feature_row()

st.subheader("Input Features (preview)")
st.dataframe(feat_df)

if model is not None:
    # Predict log(price), inverse transform
    try:
        pred_log = model.predict(feat_df)[0]
        pred_price = np.expm1(pred_log)
        st.metric("Recommended Nightly Price", f"${pred_price:,.0f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # Simple price–revenue curve using occupancy proxy and price elasticity heuristic
    # For demo: assume occupancy declines linearly with price around current occupancy input.
    prices = np.arange(candidate_min, candidate_max+1, candidate_step)
    # Elasticity heuristic: occupancy_adj = max(0, occ - k*(price - pred_price)/pred_price)
    k = 0.6
    occ_base = occupancy_rate_mean
    occ_series = np.clip(occ_base - k * (prices - max(pred_price,1)) / max(pred_price,1), 0, 1)
    revenue = prices * occ_series

    st.subheader("Revenue Curve (toy)")
    chart_df = pd.DataFrame({"price": prices, "expected_occupancy": occ_series, "expected_revenue": revenue})
    st.line_chart(chart_df.set_index("price")[["expected_revenue"]])

    # Best price (argmax)
    best_idx = np.argmax(revenue)
    best_price = prices[best_idx]
    st.info(f"Revenue-maximizing price (heuristic): **${int(best_price)}**")

st.caption("Note: Occupancy/revenue curve here is a simple heuristic. Improve it by training a demand model on calendar data.")
