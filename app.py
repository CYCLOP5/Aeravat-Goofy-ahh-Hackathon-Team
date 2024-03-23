import streamlit as st
import joblib
import pandas as pd

model_filename = 'Model/FinalModel/rff.joblib'
RF_model = joblib.load(model_filename)

# Function to preprocess input data
import pandas as pd

# Function to preprocess input data
def preprocess_input(input_data):
    # Replace 'NA' with NaN
    input_data.replace('NA', pd.NA, inplace=True)
    
    # Convert all columns to numeric (excluding 'date', 'serial_number', and 'model')
    numeric_cols = input_data.columns.difference(['date', 'serial_number', 'model'])
    input_data[numeric_cols] = input_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values with a suitable strategy (e.g., mean or median)
    input_data.fillna(input_data.median(), inplace=True)  # You can change the strategy as needed
    
    return input_data


# Function to make predictions
def predict_RUL(input_data):
    input_data = preprocess_input(input_data)
    predicted_RUL = RF_model.predict(input_data)
    return predicted_RUL

# Main function to run Streamlit app
def main():
    st.title('Hard Drive Remaining Useful Life Prediction')

    # Create input fields for user input
    st.header('Input Data')

    # Define all the features used during model training
    features = ['capacity_bytes', 'failure', 'smart_10_raw', 'smart_12_raw', 'smart_188_raw', 'smart_1_normalized', 'smart_2_normalized', 'smart_3_normalized', 'smart_4_normalized', 'smart_5_normalized', 'smart_7_normalized', 'smart_8_normalized', 'smart_9_normalized', 'smart_10_normalized', 'smart_12_normalized', 'smart_192_normalized', 'smart_193_normalized', 'smart_194_normalized', 'smart_197_normalized', 'smart_198_normalized', 'smart_199_normalized', 'smart_200_normalized', 'smart_201_normalized', 'smart_220_normalized', 'smart_222_normalized', 'smart_223_normalized', 'smart_224_normalized', 'smart_225_normalized', 'smart_226_normalized', 'smart_240_normalized', 'smart_241_normalized', 'smart_242_normalized', 'smart_250_normalized', 'smart_252_normalized', 'smart_254_normalized', 'smart_255_normalized']

    # Create input fields for each feature
    input_data = {}
    for feature in features:
        # Use appropriate input widgets based on feature types
        if feature in ['capacity_bytes', 'smart_10_raw', 'smart_12_raw', 'smart_188_raw']:  # Assuming these are numerical features
            input_data[feature] = st.number_input(feature, value=100)
        elif feature == 'failure':  # Assuming this is a binary feature
            input_data[feature] = st.radio(feature, options=[0, 1])
        else:  # Assuming other features are numerical
            input_data[feature] = st.number_input(feature, value=100)

    # Create a button to trigger predictions
    if st.button('Predict'):
        # Prepare input data as a DataFrame
        input_df = pd.DataFrame(input_data, index=[0])

        # Make predictions
        predicted_RUL = predict_RUL(input_df)

        # Display the predicted RUL
        st.header('Predicted Remaining Useful Life (RUL)')
        st.write(f'The predicted Remaining Useful Life (RUL) is: {predicted_RUL[0]} days')

if __name__ == "__main__":
    main()
