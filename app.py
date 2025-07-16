from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load("sleep_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # serves index.html from /templates

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [[
        data["total_sleep"],
        data["sleep_quality"],
        data["exercise"],
        data["caffeine"],
        data["screen_time"],
        data["work_hours"],
        data["productivity"],
        data["mood"],
        data["stress"]
    ]]
    prediction = model.predict(features)[0]
    label = "✅ Healthy Sleep" if prediction == 1 else "❌ Unhealthy Sleep"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
