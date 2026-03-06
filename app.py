import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("Loan Prediction App")

st.write("Enter applicant details below")

# Load saved files
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
preprocess = pickle.load(open("preprocess.pkl", "rb"))
final_features = pickle.load(open("final_features.pkl", "rb"))

# User Inputs

gender = st.selectbox("Gender", ["Male", "Female"])

married = st.selectbox("Married", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["0","1","2","3+"])

education = st.selectbox("Education", ["Graduate","Not Graduate"])

self_employed = st.selectbox("Self Employed", ["Yes","No"])

applicant_income = st.number_input("Applicant Income", min_value=0)

coapplicant_income = st.number_input("Coapplicant Income", min_value=0)

loan_amount = st.number_input("Loan Amount", min_value=0)

loan_amount_term = st.number_input("Loan Amount Term", min_value=0)

credit_history = st.selectbox("Credit History", [1,0])

property_area = st.selectbox("Property Area", ["Rural","Semiurban","Urban"])

# Create user input data

user_data = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area
}

user_df = pd.DataFrame([user_data])

predict_button = st.button("Predict Loan Status")

if predict_button:

    # Missing value handling
    train_modes = preprocess["train_modes"]
    LoanAmount_median = preprocess["LoanAmount_median"]
    Loan_Amount_Term_median = preprocess["Loan_Amount_Term_median"]

    cat_cols = ['Gender','Married','Dependents','Self_Employed','Credit_History']

    for col in cat_cols:
        user_df[col] = user_df[col].fillna(train_modes[col])


    user_df['LoanAmount_missing_flag'] = user_df['LoanAmount'].isnull().astype(int)

    user_df['Loan_Amount_Term_missing_flag'] = user_df['Loan_Amount_Term'].isnull().astype(int)

    user_df['LoanAmount'] = user_df['LoanAmount'].fillna(LoanAmount_median)

    user_df['Loan_Amount_Term'] = user_df['Loan_Amount_Term'].fillna(Loan_Amount_Term_median)

    user_df['Credit_History'] = user_df['Credit_History'].astype(int)


    # Feature Engineering
    user_df['ApplicantIncome_log'] = np.log1p(user_df['ApplicantIncome'])

    user_df['LoanAmount_log'] = np.log1p(user_df['LoanAmount'])

    user_df['CoapplicantIncome_log'] = np.log1p(user_df['CoapplicantIncome'])

    user_df['TotalIncome'] = user_df['ApplicantIncome'] + user_df['CoapplicantIncome']

    user_df['Income_Loan_Ratio'] = user_df['TotalIncome'] / user_df['LoanAmount']

    user_df['Loan_to_Income'] = user_df['LoanAmount'] / user_df['TotalIncome']


    # Encoding
    user_df['Married'] = user_df['Married'].map({'No':0,'Yes':1})

    user_df['Property_Area'] = user_df['Property_Area'].map({
        'Rural':0,
        'Semiurban':1,
        'Urban':2
    })

    user_df['CH_x_Property_Area'] = user_df['Credit_History'] * user_df['Property_Area']

    user_df['Gender'] = user_df['Gender'].map({'Male':1,'Female':0})

    user_df['Self_Employed'] = user_df['Self_Employed'].map({'Yes':1,'No':0})


    # One-hot encoding
    user_df = pd.get_dummies(
        user_df,
        columns=['Education','Property_Area'],
        drop_first=True
    )


    # Feature order
    user_df = user_df.reindex(columns=final_features, fill_value=0)


    # Scaling
    user_scaled = scaler.transform(user_df)


    # Prediction
    prediction = model.predict(user_scaled)
    probability = model.predict_proba(user_scaled)


    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")

    st.write("Approval Probability:", round(probability[0][1]*100,2), "%")

    st.subheader("User Input Summary")

    st.write(user_df)
