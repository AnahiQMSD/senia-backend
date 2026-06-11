from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import pickle
import traceback

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
# Cargar modelo y encoder
# ==========================
print("📦 Cargando modelo...")

modelo = tf.keras.models.load_model("modelo_lsm.keras")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("✅ Modelo cargado correctamente")


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
# Predicción
# ==========================
@app.post("/predict")
def predict(data: dict):
    try:
        # Verificar que venga sequence
        if "sequence" not in data:
            return JSONResponse(
                status_code=400, content={"error": "No se recibió 'sequence'"}
            )

        sequence = np.array(data["sequence"], dtype=np.float32)

        print("Shape recibido:", sequence.shape)

        # Validar shape esperado
        if sequence.shape != (20, 135):
            return JSONResponse(
                status_code=400,
                content={"error": f"shape incorrecto: {sequence.shape}"},
            )

        # Agregar dimensión batch
        sequence = np.expand_dims(sequence, axis=0)

        # Predicción
        pred = modelo.predict(sequence, verbose=0)

        idx = int(np.argmax(pred))
        confidence = float(np.max(pred))
        label = le.inverse_transform([idx])[0]

        print(f"Predicción: {label} | " f"Confianza: {confidence:.4f}")

        return {"label": label, "confidence": confidence}

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(status_code=500, content={"error": str(e)})
