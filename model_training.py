# model_training.py

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

def train_models_from_individual_files():
    """
    Loops through data files, trains an ARIMA model for each, and saves it.
    """
    data_dir = 'data'
    models_dir = 'models'
    
    # Create the 'models' directory if it doesn't already exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: The directory '{data_dir}' was not found. Please create it and add your XLS files.")
        return

    # Loop through each Excel file in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith((".xls", ".xlsx")) and not filename.startswith("~$"):
            file_path = os.path.join(data_dir, filename)
            
            # Extract model name from the filename (e.g., "amalner_wheat")
            model_name = os.path.splitext(filename)[0]
            print(f"Training model for: {model_name.replace('_', ' ').title()}...")

            try:
                # Use appropriate engine based on file extension
                if filename.endswith(".xlsx"):
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    df = pd.read_excel(file_path, engine='xlrd')
            except Exception as e:
                print(f"Could not read {filename}. Error: {e}")
                continue

            # --- Data Preprocessing ---
            # Check for required columns
            if 'date' not in df.columns or 'modal_price' not in df.columns:
                print(f"Skipping {model_name}: missing 'date' or 'modal_price' column.")
                continue

            try:
                df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')
                df = df.dropna(subset=['date'])
                df = df.sort_values('date').set_index('date')
                price_series = df['modal_price'].astype(float)
            except Exception as e:
                print(f"Error processing data for {model_name}: {e}")
                continue

            # Ensure there is enough data to train a model
            if len(price_series) < 30:
                print(f"Skipping {model_name} due to insufficient data.")
                continue

            # --- Train the ARIMA Model ---
            try:
                model = ARIMA(price_series, order=(5, 1, 0))
                model_fit = model.fit()
            except Exception as e:
                print(f"Error training ARIMA model for {model_name}: {e}")
                continue

            # --- Save the Trained Model ---
            model_filename = os.path.join(models_dir, f"{model_name}.pkl")
            try:
                joblib.dump(model_fit, model_filename)
                print(f"✅ Model saved to {model_filename}")
            except Exception as e:
                print(f"Error saving model for {model_name}: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Model Training from Excel Files ---")
    train_models_from_individual_files()
    print("--- Model Training Complete! ---")