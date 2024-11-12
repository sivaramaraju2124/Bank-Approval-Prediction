import streamlit as st
import pickle
import numpy as np

# Load the trained model and encoder
try:
    model_rfc = pickle.load(open('model.pkl', 'rb'))  # Load the RandomForest model
    le = pickle.load(open('le_encoder.pkl', 'rb'))  # Load the pre-fitted LabelEncoder
except FileNotFoundError:
    st.error("Model or encoder file not found. Please check file paths.")

# Streamlit app setup
st.title("Loan Approval Prediction")
st.write("Enter the details below to check if your loan application would be approved.")

# Input fields for each feature (use numeric values: 0 or 1)
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)

# Input for education level (select numeric values: 0 or 1)
education = st.selectbox("Education Level", [0, 1], format_func=lambda x: "Graduate" if x == 1 else "Not Graduate")

# Input for self-employment status (select numeric values: 0 or 1)
self_employed = st.selectbox("Are you Self-Employed?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

income_annum = st.number_input("Annual Income (in ₹)", min_value=0, step=50000)
loan_amount = st.number_input("Loan Amount Requested (in ₹)", min_value=0, step=10000)
loan_term = st.number_input("Loan Term (in months)", min_value=1, max_value=30, step=1)
cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, step=1)
total_assets_value = st.number_input("Total Assets Value (in ₹)", min_value=0, step=50000)

# Prediction button
if st.button("Predict Loan Approval"):
    if model_rfc:
        # Directly input the feature values as numeric (0 or 1) without LabelEncoder
        features = np.array([[no_of_dependents, education, self_employed, 
                              income_annum, loan_amount, loan_term, cibil_score, total_assets_value]])

        # Predict loan approval
        prediction = model_rfc.predict(features)
        
        if prediction[0] == 1:
            st.success("🎉 Congratulations! Your loan application is likely to be approved.")
        else:
            st.error("😔 Unfortunately, your loan application may not be approved.")
    else:
        st.warning("The model is not loaded correctly.")
