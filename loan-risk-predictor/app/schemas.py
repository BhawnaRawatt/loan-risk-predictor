from pydantic import BaseModel

class LoanRequest(BaseModel):
    age: int
    income: float
    loan_amount: float
    loan_term_months: int
    credit_score: int
    num_existing_loans: int
    employment_years: float
    debt_to_income: float
    has_collateral: int
    education: str
    loan_purpose: str
    region: str