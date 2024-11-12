import pandas as pd
import re
import os
import sys

def concatenate_texts(row):
    cad_text = row['CAD_TEXT'] if pd.notna(row['CAD_TEXT']) else ""
    oh_text = row['OH1_TEXT'] if pd.notna(row['OH1_TEXT']) else ""
    
    if oh_text:  # If 'OH_TEXT' is not an empty string
        return cad_text + "POLICE NARRATIVE \n\n " + oh_text
    else:
        return cad_text  # If 'OH_TEXT' is empty, return only 'CAD_TEXT'

def clean_text(df, column):
    # Define the unwanted text and regex patterns
    patterns = [
        (r"THIS DOCUMENT WAS CREATED BY AN APPLICATION THAT ISN’T LICENSED TO USE NOVAPDF. PURCHASE A LICENSE TO GENERATE PDF FILES WITHOUT THIS NOTICE.", ''),
        (r"INCIDENT REPORT PRINT .*?\n\n", '', re.DOTALL),
        (r"REDACTION DATE.*?\n\n", '', re.DOTALL)
    ]
    
    # Apply replacements
    df[column] = df[column].apply(lambda text: _apply_patterns(text, patterns))
    return df

def _apply_patterns(text, patterns):
    for pattern, replacement, *flags in patterns:
        flags = flags[0] if flags else 0
        text = re.sub(pattern, replacement, text, flags=flags)
    return text

# Map src directory
def map_project_directories():
    root_dir = os.path.abspath(os.path.join(os.getcwd(), "../."))
    print("Root Directory: ", root_dir)
    src_dir = os.path.join(root_dir,"src")
    print("Src Directory: ", src_dir)
    sys.path.append(src_dir)
    data_dir = os.path.join(root_dir,"data")
    print("Data Directory: ", data_dir)
    models_dir = os.path.join(root_dir,"models")
    print("Models Directory: ", models_dir)
    return root_dir, src_dir, data_dir, models_dir