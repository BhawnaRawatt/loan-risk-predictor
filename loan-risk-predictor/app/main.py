from fastapi import FastAPI
from app.schemas import LoanRequest
from app.predictor import predict_loan_risk

app = FastAPI(
    title="Loan Risk Predictor API",
    description="Predict loan default risk using Machine Learning",
    version="1.0.0"
)

@app.get("/")
def home():
    return{
        "message": "Loan Risk Predictor API Running"
    }

@app.post("/predict")
def predict(request: LoanRequest):
    result = predict_loan_risk(request.model_dump()
                               )
    return result


