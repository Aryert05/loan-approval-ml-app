import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Loan Approval Prediction App")

st.write("Enter applicant details below:")

# Numeric Inputs
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_History = st.selectbox("Credit History", [0, 1])

# Categorical Inputs
Gender_Male = st.selectbox("Gender (Male=1, Female=0)", [0, 1])
Married_Yes = st.selectbox("Married (Yes=1, No=0)", [0, 1])
Dependents_1 = st.selectbox("Dependents = 1", [0, 1])
Dependents_2 = st.selectbox("Dependents = 2", [0, 1])
Dependents_3_plus = st.selectbox("Dependents = 3+", [0, 1])
Education_Not_Graduate = st.selectbox("Not Graduate (1=Yes, 0=No)", [0, 1])
Self_Employed_Yes = st.selectbox("Self Employed (1=Yes, 0=No)", [0, 1])
Property_Area_Semiurban = st.selectbox("Property Area = Semiurban", [0, 1])
Property_Area_Urban = st.selectbox("Property Area = Urban", [0, 1])

if st.button("Predict Loan Status"):
    input_data = np.array([[ApplicantIncome,
                            CoapplicantIncome,
                            LoanAmount,
                            Loan_Amount_Term,
                            Credit_History,
                            Gender_Male,
                            Married_Yes,
                            Dependents_1,
                            Dependents_2,
                            Dependents_3_plus,
                            Education_Not_Graduate,
                            Self_Employed_Yes,
                            Property_Area_Semiurban,
                            Property_Area_Urban]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Loan Approved ✅ (Probability: {probability:.2f})")
    else:
        st.error(f"Loan Rejected ❌ (Probability: {probability:.2f})")
