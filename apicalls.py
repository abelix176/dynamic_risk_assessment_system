import requests
import json
import os

with open('config.json', 'r') as f:
    config = json.load(f)
output_model_path = os.path.join(config['output_model_path'])

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"


# API endpoints
prediction_endpoint = URL + f'/prediction'
scoring_endpoint = URL + f'/scoring'
summarystats_endpoint = URL + f'/summarystats'
diagnostics_endpoint = URL + f'/diagnostics'

# Call prediction endpoint
file_path = 'testdata/testdata.csv'
response1 = requests.post(prediction_endpoint, data={'file_path': file_path})

# Call scoring endpoint
response2 = requests.get(scoring_endpoint)

# Call summary statistics endpoint
response3 = requests.get(summarystats_endpoint)

# Call diagnostics endpoint
response4 = requests.get(diagnostics_endpoint)

responses = [response1, response2, response3, response4]
for response in responses:
    try:
        assert response.status_code == 200
    except AssertionError as e:
        print(f"Status code error: {response.status_code} != 200")
        raise e

# Combine outputs
predictions = response1.json()
score = response2.text
summary_stats = response3.json()
diagnostics_output = response4.json()
api_outputs = {
    'predictions': predictions,
    'score': score,
    'summary_stats': summary_stats,
    'diagnostics_output': diagnostics_output
}

# Write combined outputs to file
output_file_path = os.path.join(output_model_path, 'apireturns.txt')
with open(output_file_path, 'w') as f:
    json.dump(api_outputs, f, indent=4)



