import requests
import json
import os

with open('config.json', 'r') as f:
    config = json.load(f)
output_model_path = os.path.join(config['output_model_path'])

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/:8000"


# API endpoints
prediction_endpoint = URL + f'/prediction'
scoring_endpoint = URL + f'/scoring'
summarystats_endpoint = URL + f'/summarystats'
diagnostics_endpoint = URL + f'/diagnostics'

# Call prediction endpoint
file_path = '/testdata/testdata.csv'
response = requests.post(prediction_endpoint, data={'file_path': file_path})
predictions = response.json()

# Call scoring endpoint
response = requests.get(scoring_endpoint)
score = response.text

# Call summary statistics endpoint
response = requests.get(summarystats_endpoint)
summary_stats = response.json()

# Call diagnostics endpoint
response = requests.get(diagnostics_endpoint)
diagnostics_output = response.json()

# Combine outputs
api_outputs = {
    'predictions': predictions,
    'score': score,
    'summary_stats': summary_stats,
    'diagnostics_output': diagnostics_output
}

# Write combined outputs to file
output_file_path = os.path.join(output_model_path, 'apireturns.txt')
with open(output_file_path, 'w') as f:
    json.dump(api_outputs, f)



