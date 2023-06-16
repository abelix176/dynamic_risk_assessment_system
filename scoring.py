from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():
    # Read test data
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    # Load trained model
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Test data features
    X_test = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = test_data['exited']

    # Perform predictions
    y_pred = model.predict(X_test)
    print(f"Performing predictions.")

    # Calc F1 score
    f1 = metrics.f1_score(y_test, y_pred)
    print(f"F1 Score: {f1}")

    # Write F1 score to latestscore.txt
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(f1))
    print(f"F1 score saved to file in {model_path}")

if __name__ == '__main__':
    score_model()


