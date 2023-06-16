from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import model_predictions, dataframe_summary, check_missing_data, execution_time, outdated_packages_list
from scoring import score_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
# app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])

prediction_model = None

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    file_path = request.form.get('file_path')
    dataset = pd.read_csv(file_path)
    predictions = model_predictions(dataset)
    return jsonify(predictions)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    score_model()
    score_file_path = os.path.join(output_model_path, 'latestscore.txt')
    with open(score_file_path, 'r') as f:
        score = f.read()
    return score

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    summary_stats = dataframe_summary()
    return jsonify(summary_stats)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    timings = execution_time()
    missing_data = check_missing_data()
    module_versions = outdated_packages_list()
    diagnostics_output = {
        'timings': timings,
        'missing_data': missing_data,
        'module_versions': module_versions
    }
    return jsonify(diagnostics_output)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
