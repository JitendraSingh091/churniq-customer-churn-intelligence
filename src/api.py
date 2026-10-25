
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

model = joblib.load("models/logistic_regression_pipeline.joblib")

app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Predict customer churn using a trained ML model",
    version="1.0.0"
)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float | None = None


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_churn(customer: Customer):
    try:
        customer_data = pd.DataFrame([customer.model_dump()])

        prediction = int(model.predict(customer_data)[0])
        probability = float(model.predict_proba(customer_data)[0][1])

        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(probability, 4),
            "churn_probability_percent": round(probability * 100, 2)
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
