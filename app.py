from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('Model/rff.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.json
        
        # Preprocess the data if necessary
        # Assuming the input data is in the same format as your training data
        
        # Make prediction
        prediction = model.predict([data])[0]  # Assuming data is a list of features
        
        # Return the prediction
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
