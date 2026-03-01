from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model once at startup
model = joblib.load("aqi_model.pkl")

@app.post("/predict")
def predict(data: dict):

    # Convert incoming JSON to DataFrame
    live_df = pd.DataFrame([data])

    # Ensure correct column order
    live_df = live_df[model.feature_names_in_]

    future_predictions = []

    for i in range(3):
        pred = model.predict(live_df)[0]
        future_predictions.append(float(pred))

        # Update lag features
        live_df['AQI_lag3'] = live_df['AQI_lag2']
        live_df['AQI_lag2'] = live_df['AQI_lag1']
        live_df['AQI_lag1'] = pred
        live_df['AQI_roll3'] = (
            live_df[['AQI_lag1','AQI_lag2','AQI_lag3']].mean(axis=1)
        )

    return {
        "forecast": future_predictions
    }