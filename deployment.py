from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

if not os.path.exists(prod_deployment_path):
    os.makedirs(prod_deployment_path)


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    # Copy trained model
    model_src = os.path.join(model_path, 'trainedmodel.pkl')
    model_dest = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    shutil.copy(model_src, model_dest)
    print(f"Copied: {model_src} to: {model_dest}")

    # Copy trained model
    score_src = os.path.join(model_path, 'latestscore.txt')
    score_dest = os.path.join(prod_deployment_path, 'latestscore.txt')
    shutil.copy(score_src, score_dest)
    print(f"Copied: {score_src} to: {score_dest}")

    # Copy trained model
    ingest_src = os.path.join(dataset_csv_path, 'ingestedfiles.txt')
    ingest_dest = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    shutil.copy(ingest_src, ingest_dest)
    print(f"Copied: {ingest_src} to: {ingest_dest}")

if __name__ == '__main__':
    store_model_into_pickle()


        
        
        

