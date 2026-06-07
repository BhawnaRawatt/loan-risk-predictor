import streamlit as st
import requests

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="💰",
    layout="wide"
)



st.markdown("""
<style>

.stApp {
    background: linear-gradient(
    135deg,
    #EEF2FF 0%,
    #E0E7FF 50%,
    
    #DDD6FE 100%
    );
}
            
.header-card {
    background:white;
    padding:15px;
    border-radius:25px;
    text-align:center;
    box-shadow:0 10px 25px rgba(0,0,0,0.15);
    margin-bottom:20px;
}
.title {
    color:#1E1B4B;
    font-size:72px;
    font-weight:900;
}

.subtitle {
    color:#475569;
    font-size:22px;
}

.card {
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0 8px 20px rgba(0,0,0,0.15);
}
.stButton button {
    width:100%;
    height:60px;
    border-radius:15px;
    font-size:22px;
    font-weight:bold;
    background:linear-gradient(
        90deg,
        #2563EB,
        #4F46E5
    );
    color:white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='header-card'>
<h1 class='title'>💰 Loan Risk Predictor</h1>
<p class='subtitle'>
AI Powered Credit Risk Analysis
</p>
</div>
""", unsafe_allow_html=True)


with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 80, 30)

        loan_amount = st.number_input(
        "Loan Amount",
        min_value=1000,
        value=10000
        )
        credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650
        )

        employment_years = st.number_input(
        "Employment Years",
        min_value=0.0,
        value=5.0
        )

        has_collateral = st.selectbox(
        "Has Collateral",
        [0, 1]
        )

        loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Personal","Business","Education","Home","Auto"]
        )
    with col2:
        income = st.number_input(
        "Annual Income",
        min_value=10000,
        value=50000
        )

        loan_term_months = st.selectbox(
        "Loan Term",
        [12,24,36,48,60]
        )

        num_existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=10,
        value=1
        )  
        debt_to_income = st.slider(
        "Debt To Income Ratio",
        0.0,
        1.0,
        0.25
        )

        education = st.selectbox(
        "Education",
        ["High School","Bachelor","Master","PhD"]
        )

        region = st.selectbox(
        "Region",
        ["North","South","East","West"]
        )

    predict_btn = st.button("🚀 Predict Risk")


if predict_btn:

    payload = {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "credit_score": credit_score,
        "num_existing_loans": num_existing_loans,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "has_collateral": has_collateral,
        "education": education,
        "loan_purpose": loan_purpose,
        "region": region
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        result = response.json()
        # st.write(result)

        probability = float(
            result["default_probability"]
        )

        st.success("Prediction Completed Successfully")

        if result["risk-level"] == "High Risk":
            st.error("🔴 HIGH RISK CUSTOMER")
        else:
            st.success("🟢 LOW RISK CUSTOMER")

        st.subheader("Default Probability")

        st.progress(probability)

        st.write(
            f"Probability: {probability:.2%}"
        )

        st.markdown(f"""
        <div class='card'>

        <h2>📊 Prediction Summary</h2>

        <hr>

        <h3>Risk Level:
        {result['risk-level']}
        </h3>

        <h3>Default Probability:
        {probability:.2%}
        </h3>

        <h3>Prediction Class:
        {result['prediction']}
        </h3>

        </div>
        """, unsafe_allow_html=True)

    except Exception as e:

        st.error(
            f"API Error: {str(e)}"
        )