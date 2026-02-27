from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import base64
from io import BytesIO
from PIL import Image
import hashlib
import sqlite3
import os
import matplotlib.pyplot as plt
from datetime import datetime

app = Flask(__name__)

# -------------------------
# LOAD MODEL
# -------------------------
model = load_model("mobilenetv2_best.keras")

class_names = [
    "Apple Scab",
    "Corn Leaf Blight",
    "Potato Early Blight",
    "Tomato Leaf Mold",
    "Healthy"
]

# -------------------------
# DATABASE INIT
# -------------------------
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  disease TEXT,
                  severity REAL,
                  pest TEXT,
                  soil REAL,
                  weather TEXT,
                  yield_pred REAL,
                  hash TEXT,
                  time TEXT)''')
    conn.commit()
    conn.close()

init_db()

# -------------------------
# DISEASE DETECTION
# -------------------------
def detect_disease(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    confidence = np.max(preds) * 100
    disease = class_names[np.argmax(preds)]

    return disease, round(confidence,2)

# -------------------------
# SIMULATIONS
# -------------------------
def detect_pest(severity):
    if severity < 60:
        return "Aphids"
    return "No Pest"

def simulate_soil(img):
    brightness = np.mean(img)
    return round((brightness/255)*100,2)

def weather_risk():
    temp = 36
    if temp > 35:
        return "High Heat Risk"
    return "Normal"

def predict_yield(severity):
    return round(100 - severity*0.4,2)

def generate_hash(data):
    return hashlib.sha256(data.encode()).hexdigest()

def generate_ndvi(img):
    red = img[:,:,2].astype(float)
    green = img[:,:,1].astype(float)
    ndvi = (green - red)/(green + red + 1e-5)
    plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar()
    plt.savefig("static/ndvi.png")
    plt.close()

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.form['image_data']
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    img = Image.open(BytesIO(image_bytes))
    filepath = "static/capture.jpg"
    img.save(filepath)

    disease, severity = detect_disease(filepath)

    img_cv = np.array(img)
    pest = detect_pest(severity)
    soil = simulate_soil(img_cv)
    weather = weather_risk()
    yield_pred = predict_yield(severity)

    generate_ndvi(img_cv)

    hash_val = generate_hash(disease + str(datetime.now()))

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO predictions (disease,severity,pest,soil,weather,yield_pred,hash,time) VALUES (?,?,?,?,?,?,?,?)",
              (disease,severity,pest,soil,weather,yield_pred,hash_val,str(datetime.now())))
    conn.commit()
    conn.close()

    return render_template("result.html",
                           disease=disease,
                           severity=severity,
                           pest=pest,
                           soil=soil,
                           weather=weather,
                           yield_pred=yield_pred,
                           hash_val=hash_val)

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("dashboard.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)