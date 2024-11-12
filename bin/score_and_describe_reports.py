# General Imports
import pandas as pd
import sys
import os
import numpy as np
import pickle


# Append the src directory to sys.path and import the src modules
# Get the path to the project root (one level up from `bin/`)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

import complete_collision as cc
import utils as u

# Map the project directories
root_dir, src_dir, data_dir, models_dir = u.map_project_directories()


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