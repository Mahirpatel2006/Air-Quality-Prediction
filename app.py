from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()

# Load model at startup
model = joblib.load("aqi_model.pkl")

# Store feature order from training
FEATURES = list(model.feature_names_in_)

@app.post("/predict")
def predict(data: dict):
    try:
        # Ensure all required features are present
        missing = [feature for feature in FEATURES if feature not in data]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing}"
            )

        # Convert JSON input to numpy array in correct order
        input_array = np.array([[data[feature] for feature in FEATURES]])

        future_predictions = []

        # Get index positions for lag features once
        lag1_idx = FEATURES.index("AQI_lag1")
        lag2_idx = FEATURES.index("AQI_lag2")
        lag3_idx = FEATURES.index("AQI_lag3")
        roll_idx = FEATURES.index("AQI_roll3")

        for _ in range(3):
            pred = model.predict(input_array)[0]
            future_predictions.append(float(pred))

            # Store old lag values
            lag1 = input_array[0][lag1_idx]
            lag2 = input_array[0][lag2_idx]

            # Shift lag values
            input_array[0][lag3_idx] = lag2
            input_array[0][lag2_idx] = lag1
            input_array[0][lag1_idx] = pred

            # Update rolling mean
            input_array[0][roll_idx] = (
                input_array[0][lag1_idx] +
                input_array[0][lag2_idx] +
                input_array[0][lag3_idx]
            ) / 3

        return {"forecast": future_predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))