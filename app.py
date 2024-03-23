import streamlit as st
import pandas as pd
import joblib

model_svr = joblib.load('Model/SVR.joblib')
scaler = joblib.load('Model/scaler.pkl')

train_2_features = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21', 'sensor_2_rm', 'sensor_3_rm', 'sensor_4_rm', 'sensor_7_rm', 'sensor_8_rm', 'sensor_9_rm', 'sensor_11_rm', 'sensor_12_rm', 'sensor_13_rm', 'sensor_14_rm', 'sensor_15_rm', 'sensor_17_rm', 'sensor_20_rm', 'sensor_21_rm']

st.title('SVR Model Deployment')
st.write('Upload your data file for prediction.')

def preprocess_data(df):
    df = df[train_2_features]
    df = df.fillna(0)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df_scaled

def predict(data):
    preprocessed_data = preprocess_data(data)
    predictions = model_svr.predict(preprocessed_data)
    return predictions

uploaded_file = st.file_uploader("Upload TXT file", type=["txt"])

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file, sep=' ', header=None)
    if uploaded_data.shape[1] == len(train_2_features):
        predictions = predict(uploaded_data)
        st.write('Predictions:')
        st.write(predictions)
    else:
        st.error(f"Uploaded data has {uploaded_data.shape[1]} features, but SVR is expecting {len(train_2_features)} features as input.")
