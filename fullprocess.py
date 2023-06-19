import json
import os
import subprocess
from scoring import get_f1_score


##################Check and read new data
config_path = 'config.json'
with open(config_path, 'r') as f:
    config = json.load(f)
output_folder_path = os.path.join(config['output_folder_path'])

# Step 2: Check for New Data
deployment_path = config['prod_deployment_path']
ingested_files_path = os.path.join(deployment_path, 'ingestedfiles.txt')
input_folder_path = config['input_folder_path']

new_data_files = []
if os.path.exists(ingested_files_path):
    with open(ingested_files_path, 'r') as f:
        ingested_files = f.read().split(',')
    for file_name in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file_name)
        if file_name not in ingested_files and os.path.isfile(file_path):
            new_data_files.append(file_path)
else:
    new_data_files = [os.path.join(input_folder_path, file_name) for file_name in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, file_name))]

# Ingest new data
if new_data_files:
    subprocess.run(['python', 'ingestion.py'])



##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not new_data_files:
    print("No new data found. Exiting the deployment process.")
    exit()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
latest_score_path = os.path.join(deployment_path, 'latestscore.txt')
trained_model_path = os.path.join(deployment_path, 'trainedmodel.pkl')
new_data_path = os.path.join(output_folder_path, 'finaldata.csv')

if os.path.exists(latest_score_path) and os.path.exists(trained_model_path):
    # Read the score from the latest model
    with open(latest_score_path, 'r') as f:
        latest_score = float(f.read())
    # Get the score for the new predictions
    new_score = get_f1_score(trained_model_path, new_data_path)

    # Compare the scores
    if new_score >= latest_score:
        print("No model drift detected. Exiting the deployment process.")
        exit()
else:
    print("No previous model found. Proceeding with re-training.")


##################Re-deployment
subprocess.run(['python', 'training.py'])
subprocess.run(['python', 'scoring.py'])
subprocess.run(['python', 'deployment.py'])

##################Diagnostics and reporting
subprocess.run(['python', 'apicalls.py'])
subprocess.run(['python', 'reporting.py'])







