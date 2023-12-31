import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

#############Function for data ingestion
def merge_multiple_dataframe():
    # Initialize an empty list to store the data from each CSV file
    dataframes = []
    file_records = []

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)

            # Prepare ingest record
            dateTimeObj=datetime.now()
            thetimenow=str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)
            file_records.append([input_folder_path,filename,len(df.index),thetimenow])

    # Combine all dataframes into a single dataframe
    combined_df = pd.concat(dataframes)

    # Remove duplicate rows
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    # Save the combined dataframe as a CSV file
    output_file_path = os.path.join(output_folder_path, 'finaldata.csv')
    combined_df.to_csv(output_file_path, index=False)

    # Save ingest records to file
    ingested_files_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(ingested_files_path, 'w') as ingested_files_file:
        for sublist in file_records:
            line = ','.join(str(element) for element in sublist)
            ingested_files_file.write(line + '\n')

    # Print a confirmation message
    print(f"The combined dataframe has been saved to: {output_file_path}")
    print(f"The list of ingested filenames has been saved to: {ingested_files_path}")



if __name__ == '__main__':
    merge_multiple_dataframe()
