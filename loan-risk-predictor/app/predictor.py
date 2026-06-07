import pickle
import pandas as pd

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_loan_risk(data: dict):
    df = pd.DataFrame([data])
    df["loan_to_income_ratio"] = df["loan_amount"] / df["income"]
    df["monthly_payment_estimate"] = ( df["loan_amount"] / df["loan_term_months"] )
    df["payment_to_income_ratio"] = ( df["monthly_payment_estimate"] /(df["income"] / 12))

    if df["credit_score"][0] < 580:
        df["credit_score_band"] = 1
    elif df["credit_score"][0] < 670:
        df["credit_score_band"] = 2
    elif df["credit_score"][0] < 740:
        df["credit_score_band"] = 3
    elif df["credit_score"][0] < 800:
        df["credit_score_band"] = 4
    else:
        df["credit_score_band"] = 5

    age = df["age"][0]

    if age <= 30:
        df["age_group"] = 0
    elif age <= 40:
        df["age_group"] = 1
    elif age <= 50:
        df["age_group"] = 2
    else:
        df["age_group"] = 3

    df["is_high_risk_purpose"]    = (df["loan_purpose"] == "Personal").astype(int)
    education_map = {
        "High School": 0,
        "Bachelor": 1,
        "Master": 2,
        "PhD": 3
    }
    purpose_map = {
        "Auto": 0,
        "Business": 1,
        "Education": 2,
        "Home": 3,
        "Personal": 4
    }

    region_map = {
        "East": 0,
        "North": 1,
        "South": 2,
        "West": 3
    }


    df["education_enc"] = df["education"].map(education_map)
    df["loan_purpose_enc"] = df["loan_purpose"].map(purpose_map)
    df["region_enc"] = df["region"].map(region_map)

    features = [
        "age",
        "income",
        "loan_amount",
        "loan_term_months",
        "credit_score",
        "num_existing_loans",
        "employment_years",
        "debt_to_income",
        "has_collateral",
        "loan_to_income_ratio",
        "monthly_payment_estimate",
        "payment_to_income_ratio",
        "credit_score_band",
        "age_group",
        "is_high_risk_purpose",
        "education_enc",
        "loan_purpose_enc",
        "region_enc"
    ]

    prediction = model.predict(df[features])[0]

    probability = model.predict_proba(df[features])[0][1]

    return{
        "prediction":int(prediction),
        "default_probability": round(float(probability), 4),
        "risk-level": 
            "High Risk"
            if probability > 0.5
            else "Low Risk"
                    
    }
