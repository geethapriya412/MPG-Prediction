from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# load pipeline
model = joblib.load("mpg_pipeline.pkl")

app = FastAPI(title="Auto MPG Pipeline API")

class CarSpecs(BaseModel):
    cylinders: int
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: int
    origin: int

@app.get("/")
def home():
    return {"message": "MPG Pipeline API Running"}

@app.post("/predict")
def predict(data: CarSpecs):

    X = np.array([[
        data.cylinders,
        data.displacement,
        data.horsepower,
        data.weight,
        data.acceleration,
        data.model_year,
        data.origin
    ]])

    pred = model.predict(X)[0]

    return {
        "predicted_mpg": round(float(pred), 2)
    }
