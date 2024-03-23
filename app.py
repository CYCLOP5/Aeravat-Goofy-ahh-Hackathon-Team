import streamlit as st
import joblib
import pandas as pd
import pandas as pd

model_filename = 'Model/FinalModel/rff.joblib'
RF_model = joblib.load(model_filename)


def preprocess_input(input_data):
    input_data.replace('NA', pd.NA, inplace=True)
    numeric_cols = input_data.columns.difference(['date', 'serial_number', 'model'])
    input_data[numeric_cols] = input_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    input_data.fillna(input_data.median(), inplace=True)
    return input_data

def predict_RUL(input_data):
    input_data = preprocess_input(input_data)
    predicted_RUL = RF_model.predict(input_data)
    return predicted_RUL

def main():
    st.title('Hard Drive Remaining Useful Life Prediction')
    st.header('Input Data')

    features = ['capacity_bytes', 'failure', 'smart_10_raw', 'smart_12_raw', 'smart_188_raw', 'smart_1_normalized', 'smart_2_normalized', 'smart_3_normalized', 'smart_4_normalized', 'smart_5_normalized', 'smart_7_normalized', 'smart_8_normalized', 'smart_9_normalized', 'smart_10_normalized', 'smart_12_normalized', 'smart_192_normalized', 'smart_193_normalized', 'smart_194_normalized', 'smart_197_normalized', 'smart_198_normalized', 'smart_199_normalized', 'smart_200_normalized', 'smart_201_normalized', 'smart_220_normalized', 'smart_222_normalized', 'smart_223_normalized', 'smart_224_normalized', 'smart_225_normalized', 'smart_226_normalized', 'smart_240_normalized', 'smart_241_normalized', 'smart_242_normalized', 'smart_250_normalized', 'smart_252_normalized', 'smart_254_normalized', 'smart_255_normalized']

    input_data = {}
    for feature in features:
        if feature in ['capacity_bytes', 'smart_10_raw', 'smart_12_raw', 'smart_188_raw']:
            input_data[feature] = st.number_input(feature, value=100)
        elif feature == 'failure':
            input_data[feature] = st.radio(feature, options=[0, 1])
        else:
            input_data[feature] = st.number_input(feature, value=100)

    if st.button('Predict'):
        input_df = pd.DataFrame(input_data, index=[0])
        predicted_RUL = predict_RUL(input_df)
        st.header('Predicted Remaining Useful Life (RUL)')
        st.write(f'The predicted Remaining Useful Life (RUL) is: {predicted_RUL[0]} days')

if __name__ == "__main__":
    main()
