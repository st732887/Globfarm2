from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from ultralytics import YOLO
from werkzeug.utils import secure_filename

import os
import io
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Model paths
# -----------------------------
YOLO_MODEL_PATH = "models/best.pt"
DATA_PATH = "data/data.csv"
SCALER_X_PATH = "models/scaler_X.pkl"
SCALER_Y_PATH = "models/scaler_y.pkl"
LSTM_MODEL_PATH = "models/lstm.pth"

# -----------------------------
# Load models and data
# -----------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

data = pd.read_csv(DATA_PATH).sort_values("Price Date")

sc_x = joblib.load(SCALER_X_PATH)
sc_y = joblib.load(SCALER_Y_PATH)

# -----------------------------
# Config
# -----------------------------
SEQUENCE_LENGTH = 30
FUTURE_DAYS = 7

features = [
    "Min Price (Rs./Quintal)",
    "Max Price (Rs./Quintal)",
    "Modal Price (Rs./Quintal)",
    "temperature_C",
    "rainfall_mm"
]

# -----------------------------
# LSTM Model
# -----------------------------
class LSTMModel(nn.Module):

    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


model = LSTMModel()

model.load_state_dict(
    torch.load(
        LSTM_MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

# -----------------------------
# Prediction function
# -----------------------------
def predict_next_7_days():

    last_seq = data[features].values[-SEQUENCE_LENGTH:]

    seq = torch.tensor(
        sc_x.transform(last_seq),
        dtype=torch.float32
    ).unsqueeze(0)

    current_seq = seq.clone()

    mins = []
    maxs = []

    for _ in range(FUTURE_DAYS):

        with torch.no_grad():
            output = model(current_seq).numpy()

        pred = sc_y.inverse_transform(output)[0]

        min_price = float(pred[0])
        max_price = float(pred[1])

        mins.append(min_price)
        maxs.append(max_price)

        modal = (min_price + max_price) / 2

        last_weather = last_seq[-1, 3:]

        new_row = sc_x.transform([[
            min_price,
            max_price,
            modal,
            last_weather[0],
            last_weather[1]
        ]])

        current_seq = torch.cat(
            [
                current_seq[:, 1:, :],
                torch.tensor(new_row, dtype=torch.float32).unsqueeze(0)
            ],
            dim=1
        )

    return mins, maxs

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():

    return render_template("dashboard.html")


@app.route("/predict")
def predict():

    mins, maxs = predict_next_7_days()

    return jsonify({
        "Predicted_Min": round(mins[0], 2),
        "Predicted_Max": round(maxs[0], 2)
    })


@app.route("/plot")
def plot():

    mins, maxs = predict_next_7_days()

    days = np.arange(1, FUTURE_DAYS + 1)

    plt.figure()

    plt.plot(days, mins, marker="o", label="Min Price")
    plt.plot(days, maxs, marker="o", label="Max Price")

    plt.fill_between(days, mins, maxs, alpha=0.3)

    plt.title("7-Day Price Forecast")
    plt.xlabel("Days Ahead")
    plt.ylabel("Price (Rs./Quintal)")

    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()

    plt.savefig(buf, format="png")
    plt.close()

    buf.seek(0)

    return send_file(buf, mimetype="image/png")


@app.route("/predict_seed", methods=["POST"])
def predict_seed():

    if "image" not in request.files:

        return jsonify({"error": "No image uploaded"}), 400

    pic = request.files["image"]

    filename = secure_filename(pic.filename)

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    pic.save(filepath)

    result = yolo_model(filepath)[0]

    class_name = result.names[result.probs.top1]
    confidence = float(result.probs.top1conf)

    return jsonify({
        "class": class_name,
        "confidence": confidence,
        "image_url": f"/uploads/{filename}"
    })


@app.route("/uploads/<path:fname>")
def serve_file(fname):

    return send_from_directory(UPLOAD_FOLDER, fname)


# -----------------------------
# Run app locally
# -----------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )