from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)


@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/predict_seed", methods=["POST"])
def predict_seed():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    results = model(filepath)[0]

    class_name = results.names[results.probs.top1]
    confidence = float(results.probs.top1conf)

    return jsonify({
        "class": class_name,
        "confidence": confidence,
        "image_url": f"/uploads/{filename}"
    })


@app.route("/uploads/<path:fname>")
def serve_file(fname):
    return send_from_directory(UPLOAD_FOLDER, fname)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )