import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from mtcnn import MTCNN

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# ================= MODEL CONFIG =================
IMG_SIZE = (224, 224)
SEQ_LENGTH = 20 
MODEL_PATH = r"D:\proj\deepdetect\final_train_model.keras"

detector = MTCNN()
model = None


def load_deepdetect_model():
    global model
    
    print("Booting DEEPDETECT Engine...")

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("✅ Model Loaded Successfully!")

    except Exception as e:
        print("❌ MODEL LOAD FAILED:")
        print(e)
        model = None

    print("====================================")


def process_video_for_prediction(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "Corrupted video file."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None, "Empty video file."

    max_attempts = min(total_frames, SEQ_LENGTH * 4)
    frame_indices = np.linspace(0, total_frames - 1, max_attempts, dtype=int)

    faces_list = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detector.detect_faces(rgb_frame)

        if len(results) > 0:
            results = sorted(results, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
            x, y, w, h = results[0]['box']
            x, y, w, h = int(abs(x)), int(abs(y)), int(w), int(h)
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)

            img_h, img_w = rgb_frame.shape[:2]
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img_w, x + w + margin_x)
            y2 = min(img_h, y + h + margin_y)

            face_roi = rgb_frame[y1:y2, x1:x2]

            if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                face_resized = cv2.resize(face_roi, IMG_SIZE, interpolation=cv2.INTER_AREA)
                face_array = tf.keras.preprocessing.image.img_to_array(face_resized)
                face_array = tf.keras.applications.xception.preprocess_input(face_array)

                faces_list.append(face_array)

            if len(faces_list) >= SEQ_LENGTH:
                break

    cap.release()

    if len(faces_list) < SEQ_LENGTH:
        if len(faces_list) == 0:
            return None, "No faces detected in video."

        while len(faces_list) < SEQ_LENGTH:
            faces_list.append(faces_list[-1])

    X = np.expand_dims(np.array(faces_list, dtype=np.float32), axis=0)

    return X, "Success"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    if model is None:
        return jsonify({"error": "AI Model failed to load on server startup."}), 500

    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['video']

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return jsonify({"error": "Invalid format. Only mp4, avi, mov allowed."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(filepath)

    # Process video
    X, msg = process_video_for_prediction(filepath)

    if X is None:
        os.remove(filepath)
        return jsonify({"error": msg}), 400

    # Predict
    try:
        prediction = model.predict(X)[0][0]

        if prediction >= 0.5:
            result = "DEEPFAKE"
            confidence = round(float(prediction) * 100, 2)
        else:
            result = "REAL"
            confidence = round((1 - float(prediction)) * 100, 2)

        os.remove(filepath)

        return jsonify({
            "status": "success",
            "result": result,
            "confidence": confidence,
            "raw_score": float(prediction)
        })

    except Exception as e:
        os.remove(filepath)
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    load_deepdetect_model()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)