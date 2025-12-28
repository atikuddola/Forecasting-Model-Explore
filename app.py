import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime
# =========================
# LOAD MODEL AND FEATURES
# =========================
model = joblib.load("hourly_bike_model.pkl")
features = joblib.load("model_features.pkl")

# Load historical data
df = pd.read_csv("hour.csv")  # must have past hourly data
df = df.rename(columns={'dteday':'ds', 'cnt':'y'})
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').reset_index(drop=True)

# Compute lag features
df['y_lag1'] = df['y'].shift(1)
df['y_lag7'] = df['y'].shift(7)
df['y_roll7'] = df['y'].rolling(window=7).mean().shift(1)
df.fillna(method='bfill', inplace=True)

st.title("🚲 24-Hour Bike Rental Prediction")
st.write("Enter the date and daily weather info. Hourly weather and lag features will be calculated automatically.")

# =========================
# USER INPUT
# =========================
st.sidebar.header("Input Features")
date_input = st.sidebar.date_input("Select Date", min_value=datetime.date(2013, 1, 1),
    max_value=datetime.date(2027, 12, 31),)
season = st.sidebar.selectbox("Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)", [1,2,3,4])
yr = st.sidebar.selectbox("Year (0=2011, 1=2012)", [0,1])
mnth = date_input.month
dayofweek = date_input.weekday()
holiday = st.sidebar.selectbox("Holiday", [0,1])
workingday = st.sidebar.selectbox("Working Day", [0,1])

# Daily weather input (normalized 0-1)
daily_temp = st.sidebar.number_input("Daily Temperature (normalized 0–1)", 0.0, 1.0, 0.5, 0.01)
daily_hum = st.sidebar.number_input("Daily Humidity (normalized 0–1)", 0.0, 1.0, 0.5, 0.01)
daily_wind = st.sidebar.number_input("Daily Windspeed (normalized 0–1)", 0.0, 1.0, 0.3, 0.01)

# =========================
# HISTORICAL HOURLY WEATHER PROFILES
# =========================
# Example: create mean hourly deviation profiles from historical data
hourly_hum_profile = df.groupby(['season','mnth','hr'])['hum'].mean().unstack(level=[0,1])
hourly_wind_profile = df.groupby(['season','mnth','hr'])['windspeed'].mean().unstack(level=[0,1])

# Mock RF temp deviation (replace with your trained RF model if available)
def rf_temp_predict(df_features):
    return 0.05 * np.sin(df_features['hr'] / 24 * 2 * np.pi)

# =========================
# BUILD HOURLY DATAFRAME
# =========================
hours = list(range(24))
future_df = pd.DataFrame({'hr': hours})
future_df['season'] = season
future_df['yr'] = yr
future_df['mnth'] = mnth
future_df['holiday'] = holiday
future_df['workingday'] = workingday
future_df['dayofweek'] = dayofweek
future_df['rush_hour'] = [1 if h in [7,8,9,16,17,18] else 0 for h in hours]

# Lag features: take from last available data
last_date = df['ds'].max()
last_day = df[df['ds'] == last_date]
future_df['y_lag1'] = last_day['y'].iloc[-1] if not last_day.empty else 100
future_df['y_lag7'] = df[df['ds'] == (last_date - pd.Timedelta(days=7))]['y'].iloc[-1] \
    if not df[df['ds'] == (last_date - pd.Timedelta(days=7))].empty else 100
future_df['y_roll7'] = last_day['y'].tail(7).mean() if not last_day.empty else 100

# =========================
# CALCULATE HOURLY WEATHER
# =========================
# Temperature using RF deviation
future_df['temp'] = daily_temp + (rf_temp_predict(future_df) - rf_temp_predict(future_df).mean())

# Humidity & wind using historical profiles
# If profiles missing, fallback to daily mean
try:
    future_df['hum'] = daily_hum + (hourly_hum_profile[(season,mnth)].values - hourly_hum_profile[(season,mnth)].values.mean())
except KeyError:
    future_df['hum'] = daily_hum

try:
    future_df['windspeed'] = daily_wind + (hourly_wind_profile[(season,mnth)].values - hourly_wind_profile[(season,mnth)].values.mean())
except KeyError:
    future_df['windspeed'] = daily_wind

# =========================
# PREDICTION
# =========================
X_pred = future_df[features]

if st.button("Predict 24-Hour Rentals"):
    y_pred = model.predict(X_pred)
    y_pred = np.maximum(y_pred, 0).astype(int)

    results = pd.DataFrame({
        'Hour': range(24),
        'Predicted Rentals': y_pred
    })

    st.subheader("📊 Hourly Prediction")
    st.dataframe(results, use_container_width=True)

    st.subheader("📈 Hourly Demand Curve")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results['Hour'], results['Predicted Rentals'], marker='o', linewidth=2, label="Predicted Rentals")
    ax.set_xticks(range(24))
    ax.set_xlim(0,23)
    ax.axvspan(7,9, color='orange', alpha=0.25, label='Morning Rush (7–9)')
    ax.axvspan(16,19, color='red', alpha=0.25, label='Evening Rush (16–19)')
    ax.set_xlabel("Hour of Day (0–23)")
    ax.set_ylabel("Predicted Rentals")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.subheader("🧮 Total Rentals for the Day")
    st.success(f"Total Predicted Rentals: {results['Predicted Rentals'].sum()} bikes")
