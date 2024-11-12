# General Imports
import pandas as pd
import sys
import os
import numpy as np
import pickle
import argparse


# Append the src directory to sys.path and import the src modules
# Get the path to the project root (one level up from `bin/`)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

import complete_collision as cc
import utils as u

# Map the project directories
root_dir, src_dir, data_dir, models_dir = u.map_project_directories()

# Set up argument parsing and define the passed arguments
parser = argparse.ArgumentParser(description="Run project script with optional testing flag.")
parser.add_argument('--testing', action='store_true', help='Reduce dataframe size for testing')
args = parser.parse_args()

# Clean descriptions from the reports
training_df = pd.read_csv(os.path.join(data_dir, "processed", "training_df.csv"))
training_df['concatenated_text'] = training_df.apply(u.concatenate_texts, axis=1)
u.clean_text(training_df, 'concatenated_text')

# Load the severity model
with open(os.path.join(models_dir,'severity.pkl'), 'rb') as f:
    severity_model = pickle.load(f)

# Predict the severity of the reports
training_df['SEVERITY_PRED'] = severity_model.predict(training_df['concatenated_text'])
training_df['SEVERITY_PRED'].value_counts()

# Copy the training_df to GCAT_pred_df
GCAT_pred_df = training_df.copy()

# Convert the severity predictions to numeric
GCAT_pred_df['SEVERITY_PRED_TEXT'] = GCAT_pred_df['SEVERITY_PRED'].replace({
    0: '',
    1: 'SEVERE INJURY',
})

# Combine GCAT models predictions with the existing text
GCAT_pred_df['GCAT_PRED_TEXT'] = GCAT_pred_df['SEVERITY_PRED_TEXT']
GCAT_pred_df['GCAT_PRED_TEXT'] = GCAT_pred_df['GCAT_PRED_TEXT'].apply(lambda x: 'NONE' if x.strip() == '' else x)
GCAT_pred_df['concatenated_text'] = GCAT_pred_df['concatenated_text'] + '\n\n GCAT INDICATORS: ' + GCAT_pred_df['GCAT_PRED_TEXT']

# Save the GCAT predictions to a csv
output_path = os.path.join(data_dir, "processed", "GCAT_pred_text.csv")
GCAT_pred_df[['concatenated_text', 'BIKE_CLE_TEXT']].to_csv(output_path, index=False)
print(f"Saved GCAT predictions to {output_path}")

# Reduce the size of the dataframe for testing
if args.testing:
    print("Running in testing mode. Only the first 5 rows will be scored by GeminiAI model.")
    GCAT_pred_df = GCAT_pred_df[:5] # For testing
else:
    print("Running in normal mode. Full dataset will be scored by GeminiAI model. This may take a while.")

# Create Narratives of Collisions using GeminiBikeCleModel
GeminiBikeCleModel = cc.GenBikeCleNarrative(google_api_key="CCR_API")
GCAT_pred_df['GenAISummary'] = GCAT_pred_df['concatenated_text'].apply(lambda x: GeminiBikeCleModel.summarize(x))

# bikeCLE_input_df
GCAT_pred_df.to_csv(os.path.join(data_dir, "processed", "GenAI_df.csv"))