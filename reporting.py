import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    # Load the test data
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    
    # Get the predicted values
    predictions = model_predictions(test_data)
    
    # Get the actual values
    actual_values = test_data['exited'].values
    
    # Generate the confusion matrix
    cm = metrics.confusion_matrix(actual_values, predictions)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the plot
    confusion_matrix_path = os.path.join(output_model_path, 'confusionmatrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

if __name__ == '__main__':
    score_model()
