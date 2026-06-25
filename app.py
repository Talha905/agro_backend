from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io

# ----------------------------------------
# FastAPI App
# ----------------------------------------

app = FastAPI(
    title="AgroSaathi Plant Disease API",
    description="Plant Disease Detection using TensorFlow Lite",
    version="1.0"
)

# ----------------------------------------
# CORS (Required for Flutter Web)
# ----------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Load TFLite Model
# ----------------------------------------

interpreter = tf.lite.Interpreter(
    model_path="model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------------------
# Load Labels
# ----------------------------------------

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

labels = [None] * len(class_indices)

for label, index in class_indices.items():
    labels[index] = label

# ----------------------------------------
# Image Preprocessing
# ----------------------------------------

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image)
    image = image.astype(np.float32)

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# ----------------------------------------
# Home Route
# ----------------------------------------

@app.get("/")
def home():
    return {
        "message": "AgroSaathi Plant Disease API Running"
    }

# ----------------------------------------
# Health Check
# ----------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": len(labels),
        "input_shape": input_details[0]["shape"].tolist(),
        "output_shape": output_details[0]["shape"].tolist()
    }

# ----------------------------------------
# Prediction Endpoint
# ----------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )

        input_data = preprocess_image(image)

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        output_data = interpreter.get_tensor(
            output_details[0]["index"]
        )

        prediction_index = int(
            np.argmax(output_data)
        )

        confidence = float(
            output_data[0][prediction_index]
        )

        predicted_class = labels[
            prediction_index
        ]

        return {
            "success": True,
            "disease": predicted_class,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }