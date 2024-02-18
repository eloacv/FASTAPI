from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd

# Cargamos el modelo del archivo pickle
with open('model_Logistic.pkl', 'rb') as file:
    model, selected_features = pickle.load(file)

# Creamos una instancia de FastAPI
app = FastAPI()

# Configuración de CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verificación estado de servicio
@app.options("/predict")
def options_predict():
    return {"status": "ok"}

# Definimos la ruta para hacer predicciones
@app.post("/predict")
def predict(data: dict):
    print("Datos recibidos:", data) # Impresión de datos recibidos
    datos_pred = pd.DataFrame({key: [value] for key, value in data.items()})
    features = datos_pred[selected_features]
    prediccion = model.predict(features)
    print("Predicción:", prediccion.tolist()) # Impresión de resultado

    return {"prediction": prediccion.tolist()}
