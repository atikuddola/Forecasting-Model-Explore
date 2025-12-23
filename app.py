import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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
st.write("Enter the date and weather info. Lag features will be calculated automatically from historical data.")

# =========================
# USER INPUT
# =========================
st.sidebar.header("Input Features")
date_input = st.sidebar.date_input("Select Date")
season = st.sidebar.selectbox("Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)", [1,2,3,4])
yr = st.sidebar.selectbox("Year (0=2011, 1=2012)", [0,1])
mnth = date_input.month
dayofweek = date_input.weekday()
holiday = st.sidebar.selectbox("Holiday", [0,1])
workingday = st.sidebar.selectbox("Working Day", [0,1])
temp = st.sidebar.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity (normalized)", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed (normalized)", 0.0, 1.0, 0.3)

# =========================
# BUILD HOURLY DATAFRAME FOR SELECTED DATE
# =========================
hourly_data = []

for hr in range(24):
    rush_hour = 1 if (7 <= hr <= 9 or 16 <= hr <= 19) else 0
    
    # get previous hour value for y_lag1, same hour last week for y_lag7, rolling mean
    last_hour = df[df['ds'] < pd.Timestamp.combine(date_input, pd.Timestamp.min.time()) + pd.Timedelta(hours=hr)]
    
    if not last_hour.empty:
        y_lag1_val = last_hour['y'].iloc[-1]
        # same hour last week
        y_lag7_val = df[df['ds'] < pd.Timestamp.combine(date_input, pd.Timestamp.min.time()) + pd.Timedelta(hours=hr) - pd.Timedelta(days=7)]
        y_lag7_val = y_lag7_val['y'].iloc[-1] if not y_lag7_val.empty else last_hour['y'].mean()
        y_roll7_val = last_hour['y'].tail(7).mean()
    else:
        y_lag1_val = 100
        y_lag7_val = 120
        y_roll7_val = 110

    row = {
        'season': season,
        'yr': yr,
        'mnth': mnth,
        'hr': hr,
        'holiday': holiday,
        'workingday': workingday,
        'temp': temp,
        'hum': hum,
        'windspeed': windspeed,
        'y_lag1': y_lag1_val,
        'y_lag7': y_lag7_val,
        'y_roll7': y_roll7_val,
        'dayofweek': dayofweek,
        'rush_hour': rush_hour
    }
    hourly_data.append(row)

X_pred = pd.DataFrame(hourly_data)[features]

# =========================
# PREDICTION
# =========================
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
    fig, ax = plt.subplots()
    ax.plot(results['Hour'], results['Predicted Rentals'], marker='o')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Predicted Rentals")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("🧮 Total Rentals for the Day")
    st.success(f"Total Predicted Rentals: {results['Predicted Rentals'].sum()} bikes")
