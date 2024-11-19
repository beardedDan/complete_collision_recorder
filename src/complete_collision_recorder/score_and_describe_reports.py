# Standard Python Libraries
import os

# Third-Party Libraries
import argparse
import pandas as pd
import pickle

# Local Modules
import complete_collision as cc
import utils as u


# Step 1: Map the project directories
root_dir, src_dir, data_dir, models_dir = u.map_project_directories()

# Step 2: Set up argument parsing and define the passed arguments, if any
parser = argparse.ArgumentParser(
    description="Run project script with optional testing flag."
)
parser.add_argument(
    "--testing", action="store_true", help="Reduce dataframe size for testing"
)
args = parser.parse_args()

# Step 3: Clean text descriptions from the reports
training_df = pd.read_csv(
    os.path.join(data_dir, "processed", "training_df.csv")
)
training_df["concatenated_text"] = training_df.apply(
    u.concatenate_texts, axis=1
)

# Step 4: Load the severity model for GCAT predictions
with open(os.path.join(models_dir, "severity.pkl"), "rb") as f:
    severity_model = pickle.load(f)

# Step 5.1: Predict the severity of the reports
training_df["SEVERITY_PRED"] = severity_model.predict(
    training_df["concatenated_text"]
)
training_df["SEVERITY_PRED"].value_counts()

# Step 5.2: Placeholder for additional GCAT predictions

# Step 6: Copy the training_df to GCAT_pred_df and manipulate the numeric
# predictions to text
GCAT_pred_df = training_df.copy()
GCAT_pred_df["SEVERITY_PRED_TEXT"] = GCAT_pred_df["SEVERITY_PRED"].replace(
    {
        0: "",
        1: "SEVERE INJURY",
    }
)

# Step 7: Combine all GCAT models predictions with the existing text
GCAT_pred_df["GCAT_PRED_TEXT"] = GCAT_pred_df["SEVERITY_PRED_TEXT"]
GCAT_pred_df["GCAT_PRED_TEXT"] = GCAT_pred_df["GCAT_PRED_TEXT"].apply(
    lambda x: "NONE" if x.strip() == "" else x
)
GCAT_pred_df["concatenated_text"] = (
    GCAT_pred_df["concatenated_text"]
    + "\n\n GCAT INDICATORS: "
    + GCAT_pred_df["GCAT_PRED_TEXT"]
)

# Step 8: Save the GCAT predictions to a csv
output_path = os.path.join(data_dir, "processed", "GCAT_pred_text.csv")
GCAT_pred_df[["concatenated_text", "BIKE_CLE_TEXT"]].to_csv(
    output_path, index=False
)
print(f"Saved GCAT predictions to {output_path}")

# Step 9: Reduce the size of the dataframe if in testing mode
if args.testing:
    print(
        "Running in testing mode. Only the first 5 rows will be scored by GeminiAI model."
    )
    GCAT_pred_df = GCAT_pred_df[:5]  # For testing
else:
    print(
        "Running in normal mode. Full dataset will be scored by GeminiAI model. This may take a while."
    )

# Step 10: Create Narratives of Collisions using GeminiBikeCleModel
GeminiBikeCleModel = cc.GenBikeCleNarrative(google_api_key="CCR_API")
GCAT_pred_df["GenAISummary"] = GCAT_pred_df["concatenated_text"].apply(
    lambda x: GeminiBikeCleModel.summarize(x)
)

# Step 11: Output to GenAI_df.csv
GCAT_pred_df.to_csv(os.path.join(data_dir, "processed", "GenAI_df.csv"))
print(
    f"Saved Gemini GenAI narratives saved to {os.path.join(data_dir, 'processed', 'GenAI_df.csv')}"
)
