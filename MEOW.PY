import requests

# Sample data with column names matching the expected format in the Flask server
sample_data = [
    {
        "Date": "2016-01-01",
        "Serial_Number": "Z305B2QN",
        "Model": "ST4000DM000",
        "Feature_1": 1.97665142785814e-311,
        "Feature_2": 0,
        "Feature_3": 113,
        # Add other features accordingly
    }
]

# Send POST request to Flask server
url = 'http://localhost:5000/predict'
data = {'data': sample_data}
response = requests.post(url, json=data)

# Check the response
if response.status_code == 200:
    result = response.json()
    print("Response content:", result)  # Print the full response content
else:
    print("Error:", response.text)
