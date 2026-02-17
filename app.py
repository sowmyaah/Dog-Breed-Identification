import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

# ==============================
# 1. Flask App Setup
# ==============================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# 2. Load Model
# ==============================

MODEL_PATH = "dogbreed_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ==============================
# 3. Load ALL Class Names (AUTO)
# ==============================

CLASS_FOLDER = "dataset/train"

class_names = sorted(
    [d for d in os.listdir(CLASS_FOLDER)
     if os.path.isdir(os.path.join(CLASS_FOLDER, d))]
)

print("Total classes loaded:", len(class_names))

# ==============================
# 4. Home Page
# ==============================

@app.route("/")
def home():
    return render_template("index.html")

# ==============================
# 5. Prediction Route
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])

    predicted_class = class_names[predicted_index]

    return render_template("result.html",
                           prediction=predicted_class,
                           image_path=filepath)

# ==============================
# 6. Run App
# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=10000)




