import streamlit as st
import pandas as pd
import joblib

# Load the SVR model
model_svr = joblib.load('Model/SVR.joblib')
scaler = joblib.load('Model/scaler.pkl')  # Load the scaler used during training

# Define the features used for training
train_2_features = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21', 'sensor_2_rm', 'sensor_3_rm', 'sensor_4_rm', 'sensor_7_rm', 'sensor_8_rm', 'sensor_9_rm', 'sensor_11_rm', 'sensor_12_rm', 'sensor_13_rm', 'sensor_14_rm', 'sensor_15_rm', 'sensor_17_rm', 'sensor_20_rm', 'sensor_21_rm']

# Streamlit UI
st.title('SVR Model Deployment')
st.write('Upload your data file for prediction.')

# Function to preprocess data
def preprocess_data(df):
    # Select only the features used for training
    df = df[train_2_features]

    # Fill NaN values with 0
    df = df.fillna(0)

    # Scale the features
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df_scaled

# Function to make predictions
def predict(data):
    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Make predictions using the SVR model
    predictions = model_svr.predict(preprocessed_data)
    return predictions

# File uploader
uploaded_file = st.file_uploader("Upload TXT file", type=["txt"])

if uploaded_file is not None:
    # Read the uploaded file
    uploaded_data = pd.read_csv(uploaded_file, sep=' ', header=None)

    # Ensure that the uploaded data has the expected number of features
    if uploaded_data.shape[1] == len(train_2_features):
        # Make predictions
        predictions = predict(uploaded_data)

        # Display results
        st.write('Predictions:')
        st.write(predictions)
    else:
        st.error(f"Uploaded data has {uploaded_data.shape[1]} features, but SVR is expecting {len(train_2_features)} features as input.")
