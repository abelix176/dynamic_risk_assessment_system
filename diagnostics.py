
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import requests

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path_prod = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(dataset):
    #read the deployed model and a test dataset, calculate predictions
    # Load trained model
    with open(os.path.join(model_path_prod, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Test data features
    X_test = dataset[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]

    # Perform predictions
    y_pred = model.predict(X_test)
    return y_pred.tolist()

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics
    # Read dataset
    dataset_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    df = pd.read_csv(dataset_path)

    # Calculate summary statictics for numeric cols
    numeric_columns = df.select_dtypes(include=[np.number])
    summary_statistics = numeric_columns.describe().transpose()
    summary_statistics = summary_statistics[['mean', '50%', 'std']]

    return summary_statistics.values.tolist()

def check_missing_data():
    # Calculate missing data, N/As
    dataset_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    df = pd.read_csv(dataset_path)

    # Calculate percent of missing data in each column
    missing_data_percentage = df.isna().mean() * 100

    return missing_data_percentage.tolist()

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
     # Measure timing for data ingestion
    data_ingestion_time = timeit.timeit("os.system('python ingestion.py')", setup="import os", number=1)

    # Measure timing for model training
    model_training_time = timeit.timeit("os.system('python training.py')", setup="import os", number=1)

    return [data_ingestion_time, model_training_time]

def get_latest_version(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        latest_version = data["info"]["version"]
        return latest_version
    return None

##################Function to check dependencies
def outdated_packages_list():
    # Read installed modules from requirements.txt
    with open('requirements.txt', 'r') as f:
        requirements = f.readlines()
    
    module_info = []
    
    for requirement in requirements:
        # Extract the module name from each requirement line
        module_name = requirement.strip().split('==')[0]
        
        # Use pip show to get the currently installed version of the module
        installed_version = subprocess.check_output(['pip', 'show', module_name]).decode().split('\n')[1].split(': ')[-1]
        
        # Get the latest available version from PyPI
        latest_version = get_latest_version(module_name)
        
        module_info.append((module_name, installed_version, latest_version))
    
    # Print the module information in a table format
    print('{:<20s}{:<20s}{:<20s}'.format('Module', 'Installed Version', 'Latest Version'))
    print('-' * 60)
    
    for module in module_info:
       print('{:<20s}{:<20s}{:<20s}'.format(module[0], module[1], module[2]))


if __name__ == '__main__':
    X_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_test.drop(['corporation', 'exited'], inplace=True, axis=1)
    model_predictions(X_test)
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
