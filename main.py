from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import pickle
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("ANTES DE CARGAR MODELO")
print("Archivos en carpeta:")
print(os.listdir("."))
print("Tamaño:", os.path.getsize("modelo_lsm.keras"))
modelo = tf.keras.models.load_model("modelo_lsm.keras", compile=False)

print("DESPUÉS DE CARGAR MODELO")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("✅ Modelo cargado correctamente")


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: dict):
    try:
        if "sequence" not in data:
            return JSONResponse(
                status_code=400, content={"error": "No se recibió 'sequence'"}
            )

        sequence = np.array(data["sequence"], dtype=np.float32)

        print("Shape recibido:", sequence.shape)

        if sequence.shape != (20, 135):
            return JSONResponse(
                status_code=400,
                content={"error": f"shape incorrecto: {sequence.shape}"},
            )

        sequence = np.expand_dims(sequence, axis=0)

        pred = modelo.predict(sequence, verbose=0)

        idx = int(np.argmax(pred))
        confidence = float(np.max(pred))
        label = le.inverse_transform([idx])[0]

        print(f"Predicción: {label} | " f"Confianza: {confidence:.4f}")

        return {"label": label, "confidence": confidence}

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(status_code=500, content={"error": str(e)})
