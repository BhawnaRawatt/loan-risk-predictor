# 💰 Loan Risk Prediction System

An End-to-End Machine Learning project that predicts loan default risk using Supervised and Unsupervised Learning techniques.

Built with:

* Python
* Scikit-Learn
* FastAPI
* Streamlit
* Pandas
* NumPy
* K-Means Clustering

---

## 🚀 Project Overview

Financial institutions need to evaluate whether a customer is likely to default on a loan.

This project predicts loan risk using Machine Learning and provides a user-friendly web interface for real-time predictions.

The system includes:

* Data Generation
* Feature Engineering
* Supervised Learning Models
* Customer Segmentation
* FastAPI Backend
* Streamlit Frontend
* Model Serialization
* API Testing

---

## 🏗 Architecture

User Input (Streamlit)
↓
FastAPI API
↓
Feature Engineering
↓
Random Forest Model
↓
Risk Prediction
↓
Response Display

---

## 📊 Machine Learning Models

### Supervised Learning

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

### Unsupervised Learning

* K-Means Clustering

Customer segmentation is performed using:

* Income
* Credit Score
* Debt To Income Ratio
* Employment Years
* Loan To Income Ratio

---

## ⚙️ Feature Engineering

Custom features created:

* Loan To Income Ratio
* Monthly Payment Estimate
* Payment To Income Ratio
* Credit Score Band
* Age Group
* High Risk Purpose Flag

---

## 📁 Project Structure

```text
loan-risk-prediction-system/

├── app/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── data/
│
├── frontend/
│   └── streamlit_app.py
│
├── ml/
│   └── train_pipeline.py
│
├── models/
│   ├── best_model.pkl
│   ├── kmeans_model.pkl
│   ├── metadata.json
│   └── feature_columns.pkl
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

## 🔥 API Endpoints

### GET /

Returns API status.

### POST /predict

Predict customer loan default risk.

Sample Request:

```json
{
  "age": 35,
  "income": 50000,
  "loan_amount": 10000,
  "loan_term_months": 36,
  "credit_score": 720,
  "num_existing_loans": 1,
  "employment_years": 5,
  "debt_to_income": 0.25,
  "has_collateral": 1,
  "education": "Bachelor",
  "loan_purpose": "Home",
  "region": "North"
}
```

Sample Response:

```json
{
  "prediction": 1,
  "default_probability": 0.57,
  "risk_level": "High Risk"
}
```

---

## 🖥 Streamlit Dashboard

Features:

* User-friendly input form
* Real-time prediction
* Risk classification
* Default probability visualization
* FastAPI integration

---

## 📈 Model Evaluation

Metrics Used:

* Accuracy
* F1 Score
* ROC-AUC
* Cross Validation
* Confusion Matrix

---

## ▶️ Run Locally

### Clone Repository

```bash
git clone <repository-url>
cd loan-risk-prediction-system
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python ml/train_pipeline.py
```

### Run FastAPI

```bash
uvicorn app.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### Run Streamlit

```bash
streamlit run frontend/streamlit_app.py
```

---

## 📚 Learning Outcomes

This project demonstrates:

* Supervised Learning
* Unsupervised Learning
* Feature Engineering
* Model Evaluation
* FastAPI Development
* Streamlit Dashboard Development
* End-to-End ML Deployment

---

## 👩‍💻 Author

Bhawna Rawat

Machine Learning | Python | FastAPI | .NET Developer
