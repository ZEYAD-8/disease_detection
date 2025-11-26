from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# تحميل الموديل والسكيلر وأسماء الخصائص
with open("heart_api_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("heart_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("heart_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

@app.route("/")
def home():
    return "Heart model API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing 'features' in request."}), 400

    try:
        input_data = [features[f] for f in feature_names]
        print(input_data)
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        return jsonify({"prediction": int(prediction), "probability": round(prob, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
