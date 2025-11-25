from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)
CORS(app)

# ==========================================================
# GLOBALS
# ==========================================================
model = None
scaler = None

# ==========================================================
# FEATURE ENCODING MAPS
# ==========================================================
ENCODING_MAPS = {
    'internet_quality': {'excellent': 4, 'good': 3, 'average': 2, 'poor': 1},
    'device_availability': {'always': 4, 'mostly': 3, 'sometimes': 2, 'rarely': 1},
    'family_support': {'strong': 4, 'moderate': 3, 'limited': 2, 'none': 1},
    'participation': {'high': 4, 'medium': 3, 'low': 2, 'none': 1}
}

# ==========================================================
# TRAINING DATA
# ==========================================================
TRAINING_DATA = np.array([
    # marks, att, backlogs, assign%, internet, device, family, stress, study hrs, participation, RESULT
    [420, 95, 0, 90, 4, 4, 4, 3, 6, 4, 1],
    [380, 88, 1, 85, 3, 4, 3, 4, 5, 3, 1],
    [250, 78, 2, 70, 3, 3, 3, 5, 4, 3, 1],
    [220, 76, 2, 65, 2, 3, 2, 6, 3, 2, 1],
    [180, 70, 3, 50, 2, 2, 2, 7, 2, 2, 0],
    [150, 65, 4, 40, 1, 2, 1, 8, 2, 1, 0],
    [350, 85, 1, 80, 3, 3, 3, 4, 5, 3, 1],
    [280, 80, 2, 75, 3, 3, 3, 5, 4, 3, 1],
    [190, 72, 3, 55, 2, 2, 2, 7, 3, 2, 0],
    [410, 92, 0, 88, 4, 4, 4, 2, 6, 4, 1],
    [240, 77, 2, 68, 2, 3, 2, 6, 3, 2, 1],
    [170, 68, 4, 45, 1, 2, 1, 8, 2, 1, 0],
    [320, 82, 1, 78, 3, 3, 3, 4, 5, 3, 1],
    [260, 79, 2, 72, 3, 3, 2, 5, 4, 3, 1],
    [200, 74, 3, 60, 2, 2, 2, 6, 3, 2, 0],
    [440, 96, 0, 92, 4, 4, 4, 2, 7, 4, 1],
    [210, 75, 2, 65, 2, 3, 2, 6, 3, 2, 1],
    [160, 66, 4, 42, 1, 2, 1, 8, 2, 1, 0],
    [370, 87, 1, 83, 3, 4, 3, 3, 5, 3, 1],
    [290, 81, 2, 76, 3, 3, 3, 5, 4, 3, 1],
    [230, 78, 2, 70, 3, 3, 3, 5, 4, 2, 1],
    [195, 73, 3, 58, 2, 2, 2, 7, 3, 2, 0],
    [340, 84, 1, 79, 3, 3, 3, 4, 5, 3, 1],
    [270, 80, 2, 74, 3, 3, 3, 5, 4, 3, 1],
    [185, 71, 3, 52, 2, 2, 2, 7, 2, 2, 0],
])

# ==========================================================
# TRAINING FUNCTION
# ==========================================================
def train_model():
    """Train RandomForest model"""
    global model, scaler
    X = TRAINING_DATA[:, :-1]
    y = TRAINING_DATA[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_scaled, y)

    print("Model trained successfully.")
    return model, scaler

# ==========================================================
# ENSURE MODEL EXISTS (REQUIRED FOR VERCEL)
# ==========================================================
def ensure_model():
    global model, scaler
    if model is None or scaler is None:
        print("Model not initialized. Training now...")
        train_model()

# ==========================================================
# HELPERS
# ==========================================================
def encode_categorical_features(data):
    encoded = data.copy()
    for feature, mapping in ENCODING_MAPS.items():
        if feature in encoded:
            encoded[feature] = mapping.get(encoded[feature], 2)
    return encoded

def get_feature_vector(data):
    return [
        float(data['total_marks']),
        float(data['attendance']),
        int(data['backlogs']),
        float(data['assignment_submission']),
        int(data['internet_quality']),
        int(data['device_availability']),
        int(data['family_support']),
        float(data['stress_level']),
        float(data['study_hours']),
        int(data['participation'])
    ]

# ==========================================================
# ROUTES
# ==========================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/model-info", methods=["GET"])
def model_info():
    ensure_model()
    return jsonify({
        "model_type": "Random Forest Classifier",
        "n_estimators": 100,
        "training_samples": len(TRAINING_DATA),
        "features": 10,
        "accuracy": float(
            model.score(
                scaler.transform(TRAINING_DATA[:, :-1]),
                TRAINING_DATA[:, -1]
            ) * 100
        )
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    ensure_model()

    try:
        data = request.json
        encoded = encode_categorical_features(data)
        features = get_feature_vector(encoded)
        arr = np.array([features])
        arr_scaled = scaler.transform(arr)

        prediction = model.predict(arr_scaled)[0]
        prediction_proba = model.predict_proba(arr_scaled)[0]
        success_probability = prediction_proba[1] * 100

        # Basic risk-level logic
        if success_probability >= 75:
            risk_level = "Low Risk"
        elif success_probability >= 50:
            risk_level = "Low-Medium Risk"
        elif success_probability >= 35:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        return jsonify({
            "prediction": bool(prediction),
            "success_probability": round(success_probability, 2),
            "risk_level": risk_level
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400

# ==========================================================
# LOCAL RUN MODE
# ==========================================================
if __name__ == "__main__":
    train_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
