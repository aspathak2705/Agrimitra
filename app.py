# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Agrimitra Wheat Price Prediction",
    page_icon="🌾",
    layout="wide"
)

# --- App Title and Description ---
st.title("🌾 Agrimitra: Wheat Price Prediction for Jalgaon")
st.markdown("Select a market to predict future wheat prices and view historical trends.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Your Selections")

# Display fixed information
st.sidebar.info("**Crop:** Wheat")
st.sidebar.info("**State:** Maharashtra")
st.sidebar.info("**District:** Jalgaon")

# Create a dropdown for the fixed list of markets
jalgaon_markets = ['Bhusawal', 'Amalner', 'Chalisgaon', 'Jalgaon']
selected_market = st.sidebar.selectbox("Select a Market:", jalgaon_markets)

# Button to trigger the analysis
if st.sidebar.button("🔮 Predict & Analyze"):
    
    # --- Construct File Paths Based on User Selection ---
    safe_market_name = selected_market.replace(' ', '_').lower()
    crop_name_lower = 'wheat' # Crop is fixed
    
    model_name = f"{safe_market_name}_{crop_name_lower}"
    data_path = f"data/{model_name}.xlsx"
    model_path = f"models/{model_name}.pkl"

    # --- Check if Data and Model Files Exist Before Proceeding ---
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        st.error(f"Sorry, data or a trained model is not available for Wheat in {selected_market}.")
        st.info("Please ensure a corresponding .xlsx file is in the 'data' folder and you have run the training script.")
    else:
        try:
            # Load the data using pd.read_excel with the 'openpyxl' engine
            df = pd.read_excel(data_path, engine='openpyxl')
            df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')
        except Exception as e:
            st.error(f"Error loading or processing the data file: {e}")
            st.stop()

        try:
            # Load the pre-trained prediction model
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading the prediction model: {e}")
            st.stop()
            
        st.header(f"Results for Wheat in {selected_market}")

        # --- Prediction ---
        # Forecast prices for the next 15 days
        forecast_steps = 15
        forecast = model.forecast(steps=forecast_steps)
        
        # Create a date range for the forecasted period
        last_date = df['date'].iloc[-1]
        forecast_dates = pd.to_datetime(pd.date_range(start=last_date, periods=forecast_steps + 1)[1:])

        # Display the most immediate prediction in a metric card
        st.metric(
            label="Predicted Price for Tomorrow",
            value=f"₹ {forecast.iloc[0]:.2f} / Quintal"
        )

        # --- Data Visualization ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Price Trend Graph")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot historical data (last year for clarity)
            historical_data = df.tail(365)
            ax.plot(historical_data['date'], historical_data['modal_price'], label='Historical Prices', color='#007ACC')

            # Plot the forecasted data on the same graph
            ax.plot(forecast_dates, forecast, label='Forecasted Prices', color='#FF4B4B', linestyle='--')

            ax.set_title(f'Price Trend for Wheat in {selected_market}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (₹ per Quintal)')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("📅 Forecasted Prices")
            # Create and display a dataframe of the forecasted values
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted Price (₹)': [f"{price:.2f}" for price in forecast]
            })
            st.dataframe(forecast_df.set_index('Date'))