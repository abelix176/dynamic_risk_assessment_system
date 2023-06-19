import json
import os
import subprocess


##################Check and read new data
config_path = 'config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

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

if os.path.exists(latest_score_path) and os.path.exists(trained_model_path):
    # Read the score from the latest model
    with open(latest_score_path, 'r') as f:
        latest_score = float(f.read())

    # Make predictions using the trained model
    subprocess.run(['python', 'scoring.py'])

    # Get the score for the new predictions
    with open('score.txt', 'r') as f:
        new_score = float(f.read())

    # Compare the scores
    if new_score >= latest_score:
        print("No model drift detected. Exiting the deployment process.")
        exit()
else:
    print("No previous model found. Proceeding with re-training.")

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







