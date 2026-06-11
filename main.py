from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Cargando modelo TFLite...")

interpreter = tf.lite.Interpreter(model_path="modelo_lsm.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("✅ Modelo listo")


@app.post("/predict")
def predict(data: dict):
    try:
        sequence = np.array(data["sequence"], dtype=np.float32)

        if sequence.shape != (20, 135):
            return {"error": f"shape incorrecto: {sequence.shape}"}

        sequence = np.expand_dims(sequence, axis=0)

        interpreter.set_tensor(input_details[0]["index"], sequence)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])

        label = le.inverse_transform([np.argmax(pred)])[0]
        confidence = float(np.max(pred))

        return {"label": label, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}
