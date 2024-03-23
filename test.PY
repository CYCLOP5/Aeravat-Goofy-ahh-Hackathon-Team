import requests

sample_data = [
    {
        "Date": "2016-01-01",
        "Serial_Number": "Z305B2QN",
        "Model": "ST4000DM000",
        "Feature_1": 1.97665142785814e-311,
        "Feature_2": 0,
        "Feature_3": 113,
    }
]

url = 'http://localhost:5000/predict'
data = {'data': sample_data}
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Response content:", result)
else:
    print("Error:", response.text)
