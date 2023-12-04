import streamlit as st
import pandas as pd
import pickle

# Load the model
model = pickle.load(open('Model.sav', 'rb'))

# Function to preprocess input data
def preprocess_input(data):
    # Convert the input data into a DataFrame
    df = pd.DataFrame(data, columns=['Dependents', 'tenure', 'OnlineSecurity',
                                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                      'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

    # Manually map categorical features to numerical representations
    mapping_dict = {
        'Yes': 1,
        'No': 0,
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }

    for feature in df.columns:
        if df[feature].dtypes == 'O':
            df[feature] = df[feature].map(mapping_dict)

    return df

# Streamlit app
def main():
    st.title("Customer Churn Prediction")

    # Input form
    st.sidebar.header("User Input")
    Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure", 0, 100, 0)
    OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No"])
    OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
    DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
    TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
    Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    MonthlyCharges = st.sidebar.slider("Monthly Charges", 0.0, 500.0, 0.0)
    TotalCharges = st.sidebar.slider("Total Charges", 0.0, 10000.0, 0.0)

    # Predict button
    if st.sidebar.button("Predict"):
        input_data = [[Dependents, tenure, OnlineSecurity, OnlineBackup, DeviceProtection,
                       TechSupport, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]]

        # Preprocess input data
        input_df = preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[:, 1] * 100

        # Display prediction
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is likely to continue.")

        st.write(f"Confidence level: {round(probability[0], 2)}%")

if __name__ == "__main__":
    main()
