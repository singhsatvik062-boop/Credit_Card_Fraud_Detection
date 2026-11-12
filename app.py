
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load the trained ML model
model = joblib.load("fraud_detection_model.pkl")

# Expected features from the Kaggle dataset
FEATURES = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Credit Card Fraud Detection API is running!"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Expected a JSON object"}), 400

        if set(data.keys()) != set(FEATURES):
            return jsonify({
                "error": "Provide exactly the 30 required features",
                "features": FEATURES
            }), 400

        transaction = pd.DataFrame([data], columns=FEATURES)
        transaction = transaction.apply(
            pd.to_numeric, errors="raise"
        )

        if not transaction.notna().all().all():
            return jsonify({"error": "Invalid feature values"}), 400

        prediction = int(model.predict(transaction)[0])
        score = float(model.predict_proba(transaction)[0][1])

        return jsonify({
            "prediction": prediction,
            "result": (
                "Potential Fraud"
                if prediction == 1
                else "Likely Genuine"
            ),
            "fraud_score": round(score, 4)
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
