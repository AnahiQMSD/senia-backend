from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pickle
import tensorflow as tf
from threading import Lock
import traceback
import os

app = FastAPI()

# ==========================
# CORS
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# Rutas de prueba
# ==========================
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ==========================
# Cargar modelo
# ==========================
print("🚀 Iniciando API...")
print("PID:", os.getpid())

print("📦 Cargando modelo TFLite...")

interpreter = tf.lite.Interpreter(model_path="modelo_lsm.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("📦 Cargando LabelEncoder...")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("✅ Modelo listo")

# Lock para evitar concurrencia
interpreter_lock = Lock()


# ==========================
# Predicción
# ==========================
@app.post("/predict")
def predict(data: dict):

    try:
        # Validar entrada
        if "sequence" not in data:
            return JSONResponse(
                status_code=400, content={"error": "No se recibió 'sequence'"}
            )

        sequence = np.array(data["sequence"], dtype=np.float32)

        print("Shape recibido:", sequence.shape)

        if sequence.shape != (20, 135):
            print("❌ Shape incorrecto:", sequence.shape)

            return JSONResponse(
                status_code=400,
                content={"error": f"shape incorrecto: {sequence.shape}"},
            )

        sequence = np.expand_dims(sequence, axis=0)

        # Evitar acceso simultáneo al intérprete
        with interpreter_lock:

            interpreter.set_tensor(input_details[0]["index"], sequence)

            interpreter.invoke()

            pred = interpreter.get_tensor(output_details[0]["index"])

        idx = int(np.argmax(pred))
        label = le.inverse_transform([idx])[0]
        confidence = float(np.max(pred))

        print(f"✅ Predicción: {label} | " f"Confianza: {confidence:.4f}")

        return {"label": label, "confidence": confidence}

    except Exception as e:

        print("❌ ERROR EN /predict")
        traceback.print_exc()

        return JSONResponse(status_code=500, content={"error": str(e)})
