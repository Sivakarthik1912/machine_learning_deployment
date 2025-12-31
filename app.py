import joblib
from flask import Flask, request, jsonify
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Define paths for the model and scaler
MODEL_PATH = '/content/logistic_regression_model.joblib'
SCALER_PATH = '/content/scaler.joblib'

# Load the trained model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    # Exit or handle error appropriately in a production environment

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)

    # Expect features in a list format, e.g., {"features": [6, 148, 72, 35, 0, 33.6, 0.627, 50]}
    if "features" not in data:
        return jsonify({"error": "'features' key not found in JSON data"}), 400

    features = np.array(data['features']).reshape(1, -1)

    # Scale the input features
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)[0]
    prediction_proba = model.predict_proba(scaled_features)[0].tolist()

    # Return the prediction as JSON response
    return jsonify({
        "prediction": int(prediction),
        "probabilities": prediction_proba,
        "message": "Prediction successful"
    })

# Instructions on how to run the API
print("\nTo run this Flask API:")
print("1. Save the code above to a file named 'app.py' in the same directory as 'logistic_regression_model.joblib' and 'scaler.joblib'.")
print("2. Install Flask: `pip install Flask numpy scikit-learn joblib` (if not already installed).")
print("3. Open your terminal in that directory and run: `flask run` (or `python -m flask run`).")
print("4. The API will be available at `http://127.0.0.1:5000/`.")
print("\nTo test the API (e.g., using curl):")
print("curl -X POST -H \"Content-Type: application/json\" -d \"{\"features\": [6, 148, 72, 35, 0, 33.6, 0.627, 50]}\" http://127.0.0.1:5000/predict")
print("\nExpected output for the example input should include a prediction (0 or 1) and probabilities.")

# This block is for development purposes, remove or protect in production
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
