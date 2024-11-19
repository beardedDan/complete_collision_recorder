# Standard Python Libraries
import os

# Third-Party Libraries
import argparse
import numpy as np
import pandas as pd

# Local Modules
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

# Step 3: Load raw GCAT file
gcat_path = os.path.join(
    data_dir, "raw", "gcat", "20240924-2136-gcat-results.csv"
)
gcat_df = pd.read_csv(gcat_path)

# Clean and standardize the CAD_ID
# CRASH_YR and LOCAL_REPORT_NUMBER_ID are used to confirm CAD_ID values
gcat_df["YEAR"] = gcat_df["CRASH_YR"].astype(str).str[:4]
gcat_df["CAD_ID"] = gcat_df.apply(
    lambda row: u.clean_cad_id(row["YEAR"], row["LOCAL_REPORT_NUMBER_ID"]),
    axis=1,
)
gcat_df["INTERSECTION_IND"] = np.where(
    gcat_df["INTERSECTION_ID_CURRENT"].notnull(), "Y", "N"
)

# Step 4: Load raw BikeCLE labels
bikecle_labels_path = os.path.join(
    data_dir, "raw", "bike_cle_narratives", "callsforservice-quoted.csv"
)
bikecle_label_df = pd.read_csv(bikecle_labels_path)
bikecle_label_df.rename(
    columns={"event_number": "CAD_ID", "narrative": "BIKE_CLE_TEXT"},
    inplace=True,
)
bikecle_label_df["CAD_ID"] = bikecle_label_df["CAD_ID"].astype(str)

# Step 5: Load the processed CAD and OH1 text data
cad_text_dict = {}
oh1_text_dict = {}
cad_image_dir = os.path.join(data_dir, "processed", "cad_images")
oh1_image_dir = os.path.join(data_dir, "processed", "oh1_images")
cad_text_dict = u.load_text_into_dict(cad_image_dir)
oh1_text_dict = u.load_text_into_dict(oh1_image_dir)

# Step 6: Merge the processed OH1 and CAD data into a new dataframe called merged_text_df
cad_text_df = pd.DataFrame(
    cad_text_dict.items(), columns=["CAD_ID", "CAD_TEXT"]
)
cad_text_df["CAD_ID"] = cad_text_df["CAD_ID"].apply(u.remove_extra_cad_chars)

oh1_text_df = pd.DataFrame(
    oh1_text_dict.items(), columns=["CAD_ID", "OH1_TEXT"]
)
oh1_text_df["CAD_ID"] = oh1_text_df["CAD_ID"].apply(u.remove_extra_oh1_chars)

merged_text_df = pd.merge(
    cad_text_df,
    oh1_text_df,
    on="CAD_ID",
    how="outer",
    suffixes=("_cad", "_oh1"),
)

# Step 7: Merge the combined CAD and OH1 text data to the BikeCLE labels
merged_with_labels = pd.merge(
    merged_text_df,
    bikecle_label_df[["CAD_ID", "BIKE_CLE_TEXT"]],
    on="CAD_ID",
    how="left",
)

# Step 8: Finally, merge the joined CAD, OH1, and BikeCLE data with the
# raw GCAT data
training_df = pd.merge(
    merged_with_labels,
    gcat_df[
        [
            "CAD_ID",
            "CRASH_SEVERITY_CD",
            "INTERSECTION_IND",
            "U1_PRECRASH_ACTION_CD",
            "U1_TURN_CD",
            "U1_TYPE_OF_UNIT_CD",
            "U2_PRECRASH_ACTION_CD",
            "U2_TURN_CD",
            "U2_TYPE_OF_UNIT_CD",
        ]
    ],
    on="CAD_ID",
    how="left",
)

# Step 9: Save the training dataframe to the processed directory
training_df.to_csv(
    os.path.join(data_dir, "processed", "training_df.csv"), index=False
)
print("Training dataframe saved to data/processed/training_df.csv")
