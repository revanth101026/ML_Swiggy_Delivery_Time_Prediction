import os
import math
import pickle

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_PATH = "swiggy_demographic.csv"
MODEL_PATH = "swiggy_delivery_time_model.pkl"
IMAGE_PATH = "images.jpeg"
IMAGE_WIDTH = 140  # pixels - change this to make the logo bigger/smaller
TEST_RMSE = 4.0  # test RMSE (minutes) of the final model in Swiggy_Project.ipynb

# Feature order used while training (X in the notebook, incl. engineered features)
FEATURES = [
    "age", "ratings", "weather", "traffic", "vehicle_condition",
    "type_of_order", "type_of_vehicle", "multiple_deliveries",
    "festival", "city_type", "city_name",
    "order_day", "order_month", "order_day_of_week",
    "is_weekend", "pickup_time_minutes",
    "order_time_hour", "order_time_of_day", "distance",
    "is_rush", "is_peak_hour",
]

TRAFFIC_ORDER = ["low", "medium", "high", "jam"]

st.set_page_config(page_title="Swiggy Delivery Time Prediction", layout="wide")


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    return data


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def build_lookups(data):
    """Derive hour -> time-of-day bucket and weekday -> is_weekend from the data,
    so the app never contradicts the labels the model was trained on."""
    hours = data.dropna(subset=["order_time_hour", "order_time_of_day"]).copy()
    hours["hour_int"] = hours["order_time_hour"].astype(int)
    hour_to_bucket = (
        hours.groupby("hour_int")["order_time_of_day"]
        .agg(lambda s: s.mode().iat[0])
        .to_dict()
    )
    days = data.dropna(subset=["order_day_of_week", "is_weekend"])
    day_to_weekend = (
        days.groupby("order_day_of_week")["is_weekend"]
        .agg(lambda s: int(s.mode().iat[0]))
        .to_dict()
    )
    return hour_to_bucket, day_to_weekend


def used_columns(model):
    """Columns the fitted preprocessing step actually uses (others are dropped)."""
    try:
        prep = model.named_steps["Data Preprocessing"]
        cols = set()
        for name, _, c in prep.transformers_:
            if name == "remainder":  # columns dropped by the ColumnTransformer
                continue
            cols.update(list(c))
        return cols
    except Exception:
        return set(FEATURES)


def options(data, col):
    return sorted(data[col].dropna().unique().tolist())


def mode_of(data, col):
    return data[col].mode().iat[0]


# ----------------------------------------------------------------------
# Load resources
# ----------------------------------------------------------------------
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Put it in the same folder as app.py.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{MODEL_PATH}`. Put the pickle saved from the notebook "
             "in the same folder as app.py.")
    st.stop()

hour_to_bucket, day_to_weekend = build_lookups(df)
used = used_columns(model)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>Swiggy Delivery Time Prediction</h1>",
            unsafe_allow_html=True)

if os.path.exists(IMAGE_PATH):
    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        st.image(IMAGE_PATH, width=IMAGE_WIDTH)

st.markdown("### Machine Learning Powered- Swiggy Delivery Time Predictor")

# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

# ---- 1. Rider details ----
with col1:
    st.subheader("Rider Details")

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age = st.number_input("Rider Age", min_value=age_min, max_value=age_max,
                          value=min(max(28, age_min), age_max), step=1)

    r_min, r_max = float(df["ratings"].min()), float(df["ratings"].max())
    ratings = st.slider("Rider Ratings", min_value=r_min, max_value=r_max,
                        value=min(max(4.7, r_min), r_max), step=0.1)

    vehicle_condition = st.selectbox(
        "Vehicle Condition", [int(v) for v in options(df, "vehicle_condition")],
        help="0 = poor, 1 = average, 2 = good (as per the dataset description)")

    type_of_vehicle = st.selectbox("Type of Vehicle", options(df, "type_of_vehicle"))

# ---- 2. Location / conditions ----
with col2:
    st.subheader("Location & Conditions")

    if "city_name" in used:
        city_name = st.selectbox("City Name", options(df, "city_name"))
    else:
        city_name = mode_of(df, "city_name")  # not used by the trained model

    city_type = st.selectbox("City Type", options(df, "city_type"))
    weather = st.selectbox("Weather", options(df, "weather"))

    traffic_opts = [t for t in TRAFFIC_ORDER if t in set(df["traffic"].dropna())] \
        or options(df, "traffic")
    traffic = st.selectbox("Traffic", traffic_opts)

    d_min = max(math.floor(df["distance"].min()), 1)
    d_max = math.ceil(df["distance"].max())
    distance = st.number_input("Distance (km)", min_value=float(d_min),
                               max_value=float(d_max),
                               value=float(round(df["distance"].median())), step=0.5)

# ---- 3. Order details ----
with col3:
    st.subheader("Order Details")

    type_of_order = st.selectbox("Type of Order", options(df, "type_of_order"))
    multiple_deliveries = st.selectbox(
        "Multiple Deliveries", [int(v) for v in options(df, "multiple_deliveries")])
    festival = st.selectbox("Festival", options(df, "festival"))

    date_min = df["order_date"].min().date()
    date_max = df["order_date"].max().date()
    order_date = st.date_input("Order Date", value=date_min + (date_max - date_min) / 2,
                               min_value=date_min, max_value=date_max)

    p_min, p_max = int(df["pickup_time_minutes"].min()), int(df["pickup_time_minutes"].max())
    pickup_time_minutes = st.number_input(
        "Pickup Time (minutes)", min_value=p_min, max_value=p_max,
        value=min(max(10, p_min), p_max), step=1,
        help="Time the restaurant takes to prepare the order")

    order_time_hour = st.number_input("Order Time Hour (0-23)", min_value=0,
                                      max_value=23, value=18, step=1)

# ----------------------------------------------------------------------
# Derived features (must match the notebook exactly)
# ----------------------------------------------------------------------
order_day = order_date.day
order_month = order_date.month
order_day_of_week = order_date.strftime("%A").lower()
is_weekend = int(day_to_weekend.get(order_day_of_week, 1 if order_date.weekday() >= 5 else 0))
order_time_of_day = hour_to_bucket.get(int(order_time_hour), mode_of(df, "order_time_of_day"))

# Engineered features from the notebook
is_rush = int(traffic == "jam" and is_weekend == 1)
is_peak_hour = int(18 <= order_time_hour <= 22)

st.caption(
    f"Derived automatically → day: **{order_day_of_week.title()}**, "
    f"weekend: **{'Yes' if is_weekend else 'No'}**, "
    f"time of day: **{order_time_of_day}**, "
    f"peak hour: **{'Yes' if is_peak_hour else 'No'}**, "
    f"rush: **{'Yes' if is_rush else 'No'}**"
)

# ----------------------------------------------------------------------
# Build model input
# ----------------------------------------------------------------------
input_df = pd.DataFrame([{
    "age": age,
    "ratings": ratings,
    "weather": weather,
    "traffic": traffic,
    "vehicle_condition": vehicle_condition,
    "type_of_order": type_of_order,
    "type_of_vehicle": type_of_vehicle,
    "multiple_deliveries": multiple_deliveries,
    "festival": festival,
    "city_type": city_type,
    "city_name": city_name,
    "order_day": order_day,
    "order_month": order_month,
    "order_day_of_week": order_day_of_week,
    "is_weekend": is_weekend,
    "pickup_time_minutes": pickup_time_minutes,
    "order_time_hour": order_time_hour,
    "order_time_of_day": order_time_of_day,
    "distance": distance,
    "is_rush": is_rush,
    "is_peak_hour": is_peak_hour,
}])[FEATURES]

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if st.button("Predict Delivery Time", type="primary"):
    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    prediction = max(prediction, 0.0)
    st.success(f"Estimated Delivery Time: {prediction:.1f} minutes ⏱️")
    st.text(f'Distance: {input_df['distance'][0]} km')
    st.caption(f"Typical model error is about ±{TEST_RMSE:.0f} minutes "
               f"(likely range {max(prediction - TEST_RMSE, 0):.0f}–{prediction + TEST_RMSE:.0f} min).")

    with st.expander("Inputs sent to the model"):
        st.table(input_df.T.astype(str).rename(columns={0: "value"}))