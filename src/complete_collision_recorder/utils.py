import pandas as pd
import re
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Evaluation Imports
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
    roc_auc_score,
)


def concatenate_texts(row):
    """
    Concatenate the 'CAD_TEXT' and 'OH1_TEXT' columns.

    Parameters:
        row (Series): The row containing the text data.

    Returns:
        cad_text (str): The concatenated text data.
    """
    cad_text = row["CAD_TEXT"] if pd.notna(row["CAD_TEXT"]) else ""
    oh_text = row["OH1_TEXT"] if pd.notna(row["OH1_TEXT"]) else ""

    if oh_text:  # If 'OH_TEXT' is not an empty string
        return cad_text + "POLICE NARRATIVE \n\n " + oh_text
    else:  # If it is empty, return only 'CAD_TEXT'
        return cad_text


def clean_text(df, column):
    """
    Clean the text data by removing unwanted patterns.

    Parameters:
        df (DataFrame): The DataFrame containing the text data.
        column (str): The column containing the text data.

    Returns:
        df (DataFrame): The DataFrame with cleaned text data.
    """
    patterns = [
        (
            r"THIS DOCUMENT WAS CREATED BY AN APPLICATION THAT ISN’T LICENSED TO USE NOVAPDF. PURCHASE A LICENSE TO GENERATE PDF FILES WITHOUT THIS NOTICE.",
            "",
        ),
        (r"INCIDENT REPORT PRINT .*?\n\n", "", re.DOTALL),
        (r"REDACTION DATE.*?\n\n", "", re.DOTALL),
    ]

    # Apply replacements
    df[column] = df[column].apply(lambda text: _apply_patterns(text, patterns))
    return df


def _apply_patterns(text, patterns):
    for pattern, replacement, *flags in patterns:
        flags = flags[0] if flags else 0
        text = re.sub(pattern, replacement, text, flags=flags)
    return text


def evaluate_confusion_matrix(y_true, model_best, X_test):
    """
    Evaluate the model using the confusion matrix, F1 score, and ROC AUC score.

    Parameters:
        y_true (array): The true labels.
        model_best (object): The trained model.
        X_test (array): The test data.

    Returns:
        None - Displays the confusion matrix and scores.
    """
    y_train_pred = model_best.predict(X_test)
    print("F1 score: ", f1_score(y_true, y_train_pred, average="weighted"))
    print(
        "ROC AUC score: ",
        roc_auc_score(y_true, y_train_pred, average="weighted"),
    )
    print(classification_report(y_true, y_train_pred))
    conf_matrix = confusion_matrix(y_true, y_train_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


# Map src directory
def map_project_directories(notebook=False):
    """
    Maps the project directories

    Parameters:
        notebook (bool): True if running in a notebook environment, False otherwise.

    Returns:
        tuple: root_dir, src_dir, data_dir, models_dir
    """
    if notebook:
        # Adjust paths for notebook environment
        # src_dir = os.path.abspath('../src/complete_collision_recorder/')
        src_dir = os.path.abspath(
            os.path.join(
                os.getcwd(), "../", "src", "complete_collision_recorder"
            )
        )
        root_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
        data_dir = os.path.join(root_dir, "data")
        models_dir = os.path.join(root_dir, "models")

    else:
        # Deployment environment
        src_dir = os.path.abspath(os.path.join(os.getcwd(), "../."))
        root_dir = os.path.abspath(src_dir + "/../")
        data_dir = os.path.join(root_dir, "data")
        models_dir = os.path.join(root_dir, "models")

    sys.path.append(src_dir)
    return root_dir, src_dir, data_dir, models_dir


def clean_cad_id(year, id_num):
    """
    Clean and standardize the CAD_ID based on predefined patterns.

    Parameters:
    year (str): The year associated with the record.
    id_num (str): The original ID number to be cleaned.

    Returns:
    str: The cleaned and standardized CAD_ID.
    """
    patterns = {
        "year-hyphen": r"^\d{4}-\d{5,}",
        "year_no_hyphen": r"^\d{10}$",
        "2digit-year-hyphen": r"^\d{2}-\d{5,}",
        "special": r"^\d{2}-\d{4}-\d{2}",
        "miscellaneous": r".+",
        "rpt_num_only": r"^\d{5,}",
    }

    if pd.isna(id_num):
        return None  # Return None for missing or NaN values

    id_num = str(id_num)  # Ensure id_num is a string

    if re.match(patterns["year-hyphen"], id_num):
        parts = id_num.split("-", 1)
        new_id = parts[0] + parts[1].zfill(8)  # Pad second part to 5 digits
    elif re.match(patterns["year_no_hyphen"], id_num):

        part1 = id_num[:4]
        part2 = id_num[4:]
        new_id = part1 + part2.zfill(8)
    elif re.match(patterns["2digit-year-hyphen"], id_num):
        parts = id_num.split("-", 1)
        new_id = parts[0] + parts[1].zfill(
            8
        )  # Prepend '20' and pad to 5 digits
    elif re.match(patterns["rpt_num_only"], id_num):
        new_id = year + id_num.zfill(8)  # Concatenate year and padded ID
    elif re.match(patterns["special"], id_num):
        new_id = id_num.replace("-", "")  # Remove hyphens
    else:
        new_id = id_num.replace(
            "-", ""
        )  # Remove hyphens for miscellaneous cases

    return new_id


def remove_extra_cad_chars(cad_id):
    return cad_id[:12]


def remove_extra_oh1_chars(cad_id):
    return "20" + cad_id[16:]


def read_text_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def load_text_into_dict(image_dir):
    """
    Read all text files from the given directory and store the
    content in a dictionary.

    Parameters:
    image_dir (str): The directory containing the text files.

    Returns:
    dict: A dictionary with incident IDs as keys and file contents as values.
    """
    text_dict = {}
    with ThreadPoolExecutor() as executor:
        folder_paths = [
            os.path.join(image_dir, folder)
            for folder in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, folder))
        ]
        file_paths = []
        for folder in folder_paths:
            for file in os.listdir(folder):
                if file.endswith(".txt"):
                    file_paths.append(os.path.join(folder, file))
        results = executor.map(read_text_file, file_paths)
        for file_path, content in zip(file_paths, results):
            incident_id = os.path.basename(file_path).split(".")[0]
            text_dict[incident_id] = content
    return text_dict
