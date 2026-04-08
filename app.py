from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("mpg_pipeline.pkl")



# Input schema
class CarFeatures(BaseModel):
    cylinders: int
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: int
    origin: int

@app.get("/")
def home():
    return {"message": "Auto MPG Prediction API"}

@app.post("/predict")
def predict(data: CarFeatures):
    features = np.array([[ 
        data.cylinders,
        data.displacement,
        data.horsepower,
        data.weight,
        data.acceleration,
        data.model_year,
        data.origin
    ]])

    prediction = model.predict(features)[0]

    return {
        "predicted_mpg": round(prediction, 2)
    }