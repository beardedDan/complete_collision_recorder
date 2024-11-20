# Standard Python Libraries
import csv
import logging
import os
import re
import requests
import time
import warnings
import zipfile


# Third-Party Libraries


# Visualization and Data manipulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# OCR
import cv2
from pdf2image import convert_from_path
import pytesseract

# Preprocessing and Feature Extraction
import nltk
import spacy
from nltk.stem import PorterStemmer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Model Selection and Evaluation
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Google Gemini GenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv


# Set up the logger
warnings.filterwarnings(
    "ignore",
    message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None",
)

# Load environmental variables
load_dotenv()


class PdfTextExtraction:
    """
    Class for extracting and processing text from PDF documents.
    This corresponds to Functional Requirements 01, 02, and 03 in
    the System Design.

    For more details, refer to the README.md or System Design document:
    /docs/system_design_and_functional_nonfunctional_requirements.md.

    Attributes
    ----------
    nlp : spacy.lang.en.English
        SpaCy NLP object for named entity recognition (NER) and name
        identification/removal.
    name_pattern : re.Pattern
        Regular expression pattern to identify valid first names.
    ssa_url : str
        URL pointing to the Social Security Administration (SSA)
        dataset for common names.
    unique_names : list
        List of common names derived from the SSA dataset.
    logger : logging.Logger
        Logger object to log information and errors.

    AI Highlights
    ----------
    - AI Ethics and Privacy: Personal names are redacted in two passes:
    - NLP Named Entity Recognition (NER) using AI functionality from SpaCy 
    - NLP NER using a statistical approach in the 'create_common_name_dataset'
    metehod using Social Security Administration (SSA) data.
    - Machine Vision Canny Edge Detection in the 'process_image' method
    for defining bounding boxes and matching text regions to a template.
    - Machine Visions Optical Character Recognition (OCR) in the
    'process_cad_pdfs_in_folder' and 'process_oh1_pdfs_in_folder' methods
    """

    def __init__(
        self,
        data_directory="../../data/lookup/ssa_names",
        num_top_names=2000,
        min_year=1880,
        max_year=2023,
        output_file="../../data/lookup/common_names.csv",
    ):
        """
        Initialize the PdfTextExtraction class. This method sets up the
        environment for text extraction, including loading necessary models,
        defining the name pattern, and setting up logging.

        Parameters
        ----------
        data_directory : str, optional
            Directory containing the SSA names data.
            Default is "/data/lookup/ssa_names".
        num_top_names : int, optional
            Number of top names to consider from the SSA dataset.
        min_year : int, optional
            The minimum year to filter the SSA names.
        max_year : int, optional
            The maximum year to filter the SSA names.
        output_file : str, optional
            Path to save the generated list of common names.
            Default is "/data/lookup/common_names.csv".
        """

        # Set up logging
        root_directory = os.path.abspath(os.path.join(os.getcwd(), "../."))
        log_directory = os.path.join(root_directory, "logs")
        os.makedirs(log_directory, exist_ok=True)
        log_file_path = os.path.join(log_directory, "pdf_text_extraction.log")

        # For NER and name removal, the use of SpaCy Small English model and a
        # custom list of Social Security Administration (SSA) names is used.
        # Import SpaCy
        self.nlp = spacy.load("en_core_web_trf")

        # Define a valid first name pattern and link to the SSA dataset
        self.name_pattern = re.compile(r"^[a-zA-Z]+$")
        self.ssa_url = "https://www.ssa.gov/oact/babynames/names.zip"
        self.unique_names = self.create_common_name_dataset(
            data_directory, output_file, num_top_names, min_year, max_year
        )

        # Logging Reference: https://realpython.com/python-logging/
        self.logger = logging.getLogger("PdfTextExtraction")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(
            log_file_path, mode="a", encoding="utf-8"
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def is_valid_name(self, name):
        return (
            isinstance(name, str)
            and name.strip() != ""
            and self.name_pattern.match(name)
        )

    def download_and_unzip_data(self, zip_url, target_directory):
        """
        Download a ZIP file from a URL and extract its contents to a
        target directory.

        Parameters
        ----------
        zip_url : str
            The URL of the ZIP file to download.
        target_directory : str
            The directory where the ZIP file will be extracted.

        Returns
        -------
        None
            The method does not return any value. It extracts the contents of
            the ZIP file into the specified target directory.
        """
        os.makedirs(target_directory, exist_ok=True)
        zip_file_path = os.path.join(target_directory, "names.zip")
        response = requests.get(zip_url)
        with open(zip_file_path, "wb") as zip_file:
            zip_file.write(response.content)
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(target_directory)
        os.remove(zip_file_path)

    def create_common_name_dataset(
        self,
        data_directory="../../data/lookup/ssa_names",
        output_file="../../data/lookup/common_names.csv",
        num_top_names=1000,
        min_year=1943,
        max_year=2023,
    ):
        """
        Creates a dataset of common names and saves it to a CSV file.

        Parameters
        ----------
        data_directory : str, optional
            Directory containing the SSA names data.
        output_file : str, optional
            The name of the output CSV file.
        num_top_names : int, optional
            Minimum count threshold for names to be included.
        min_year : int, optional
            Starting year for the dataset.
        max_year : int, optional
            Ending year for the dataset.

        Returns
        -------
        list of str
            A sorted list of common names that meet the threshold criteria.

        Notes
        -----
        This function downloads and processes data from the SSA, filtering
        for names with a count greater than the specified `num_top_names`
        threshold, and saves the resulting names to a CSV file.

        """

        global UNIQUE_NAMES_LIST
        self.num_top_names = num_top_names
        self.min_year = min_year
        self.max_year = max_year

        unique_names = set()
        self.download_and_unzip_data(self.ssa_url, data_directory)

        for year in range(self.min_year, self.max_year):
            filename = os.path.join(data_directory, f"yob{year}.txt")
            try:
                with open(filename, "r") as file:
                    for line in file:
                        name, gender, count = line.strip().split(",")
                        name = name
                        if int(
                            count
                        ) > self.num_top_names and self.is_valid_name(name):
                            unique_names.add(name)

                        else:
                            pass
            except FileNotFoundError:
                print(f"File {filename} not found.")

        # Write to CSV
        try:
            with open(output_file, "w", newline="") as csvfile:
                fieldnames = ["Name"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([{"Name": name} for name in unique_names])
            print(f"SSA Dataset successfully written to {output_file}.")
            return sorted(list(unique_names))
        except Exception as e:
            print(f"Error writing SSA names to CSV file: {e}")

    def process_cad_pdfs_in_folder(
        self, folder_path, output_base_folder, dpi=300
    ):
        """
        Processes CAD PDF files in a specified folder by converting them
        to images, extracting text with OCR, and saving the results to
        output folders.

        Parameters
        ----------
        folder_path : str
            Path to the CAD PDF files folder.
        output_base_folder : str
            Path to where the processed files will be saved.
        dpi : int, optional
            The DPI (dots per inch) setting used for image conversion.

        Returns
        -------
        None
            This function does not return any value but saves the
            processed results in subfolders within `output_base_folder`.

        Notes
        -----
        - The function processes all PDF files in `folder_path`,
        excluding the first page of each PDF (assumed to be a cover page).
        - Converted image files are used for OCR to extract text, which is
        then saved as a `.txt` file in the respective subfolder within
        `output_base_folder`.
        - Any errors or non-PDF files are logged.
        """

        self.logger.info(
            f"Starting PDF processing of CAD files in folder: {folder_path}"
        )

        for pdf_file in os.listdir(folder_path):
            if pdf_file.endswith(".pdf"):
                cad_id = os.path.splitext(pdf_file)[0]
                pdf_path = os.path.join(folder_path, pdf_file)
                output_folder = os.path.join(output_base_folder, cad_id)
                os.makedirs(output_folder, exist_ok=True)

                self.logger.info(f"Processing CAD PDF: {pdf_file}")

                try:
                    image_files = self.convert_all_page_to_image(
                        pdf_path, output_folder, dpi
                    )

                    if image_files:
                        extracted_text = []
                        for i, image_file in enumerate(image_files):
                            # Do not process the first page of CAD files
                            # The first page is a cover page
                            if i == 0:
                                continue
                            page_text = self.process_ocr(
                                image_file, output_folder, cad_id
                            )
                            extracted_text.append(page_text)

                        full_text = "\n\n".join(extracted_text)
                        output_text_file = os.path.join(
                            output_folder, f"{cad_id}_ocr.txt"
                        )

                        with open(output_text_file, "w") as text_file:
                            text_file.write(full_text)

                    else:
                        self.logger.warning(
                            f"Could not convert from: {pdf_file}"
                        )
                except Exception as e:
                    self.logger.error(f"Error processing file {pdf_file}: {e}")
            else:
                self.logger.info(f"Non-PDF file {pdf_file} found")

        self.logger.info(f"All files processed from {folder_path}")

    def process_oh1_pdfs_in_folder(
        self, folder_path, output_base_folder, template_path, dpi=300
    ):
        """
        Processes OH1 PDF files in a specified folder by converting them to
        images, extracting text with OCR, and saving the results to output
        folders.

        Parameters
        ----------
        folder_path : str
            Path to the OH1 PDF files folder.
        output_base_folder : str
            Path to where the processed files will be saved.
        dpi : int, optional
            The DPI (dots per inch) setting used for image conversion.

        Returns
        -------
        None
            This function does not return any value but saves the processed
            results in subfolders within `output_base_folder`.

        Notes
        -----
        - The function processes all PDF files in `folder_path`, excluding the
        first page of each PDF (assumed to be a cover page).
        - Converted image files are used for OCR to extract text, which is then
        saved as a `.txt` file in the respective subfolder within
        `output_base_folder`.
        - Any errors or non-PDF files are logged.
        """

        self.logger.info(
            f"Starting PDF processing of OH1 in folder: {folder_path}"
        )

        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            self.logger.error(f"Template image not found at {template_path}")
            raise FileNotFoundError(
                f"Template image not found {template_path}"
            )
        self.logger.info(f"Template loaded from {template_path}")

        for pdf_file in os.listdir(folder_path):
            if pdf_file.endswith(".pdf"):
                cad_id = os.path.splitext(pdf_file)[0]
                pdf_path = os.path.join(folder_path, pdf_file)
                output_folder = os.path.join(output_base_folder, cad_id)
                os.makedirs(output_folder, exist_ok=True)

                self.logger.info(f"Processing OH1 PDF: {pdf_file}")

                try:
                    image_file = self.convert_first_page_to_image(
                        pdf_path, output_folder, dpi
                    )
                    if image_file:
                        self.process_image(
                            image_file, template, output_folder, cad_id
                        )
                    else:
                        self.logger.warning(
                            f"Could not convert from: {pdf_file}"
                        )
                except Exception as e:
                    self.logger.error(f"Error processing file {pdf_file}: {e}")

        self.logger.info(f"All files processed from {folder_path}")

    def convert_all_page_to_image(self, pdf_path, output_folder, dpi=300):
        """
        Converts all pages of a PDF file to images and saves them as PNG files.

        Parameters
        ----------
        pdf_path : str
            The path to the PDF file to be converted.
        output_folder : str
            The folder where the resulting images will be saved.
        dpi : int, optional
            The DPI (dots per inch) for the image conversion.

        Returns
        -------
        list of str
            A list of file paths to the saved images (one per page).

        Notes
        -----
        - Each page of the PDF is converted to a separate PNG image.
        - The images are saved with filenames in the format
        `page_<page_number>.png`.
        """
        images = convert_from_path(pdf_path, dpi=dpi)  # Convert all pages

        image_paths = []

        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"page_{i+1}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        return image_paths

    def convert_first_page_to_image(self, pdf_path, output_folder, dpi=300):
        """
        Converts the first page of a PDF file to an image and saves it
        as a PNG file.

        Parameters
        ----------
        pdf_path : str
            The path to the PDF file to be converted.
        output_folder : str
            The folder where the resulting image will be saved.
        dpi : int, optional
            The DPI (dots per inch) for the image conversion. Default is 300.

        Returns
        -------
        str or None
            The file path of the saved first page image if successful,
            or None if the conversion fails.

        Notes
        -----
        - Only the first page of the PDF is converted to an image.
        - The resulting image is saved with the filename `page_1.png`.
        """
        images = convert_from_path(
            pdf_path, dpi=dpi, first_page=1, last_page=1
        )  # Limit to first page only
        if images:
            image_path = os.path.join(output_folder, "page_1.png")
            images[0].save(image_path, "PNG")
            return image_path
        return None

    def process_image(self, image_path, template, output_folder, cad_id):
        """
        Processes a single image by applying various image processing
        techniques: resizing, blurring, edge detection, contour detection,
        bounding box matching, text extraction.

        Parameters
        ----------
        image_path : str
            The path to the image file to be processed.
        template : np.array
            The template image used for bounding box matching.
        output_folder : str
            The folder where the output images and extracted text will be saved.
        cad_id : str
            The identifier for the current file, used in naming output files.

        Returns
        -------
        None
            This function does not return a value but saves processed results
            (extracted images and text) to the `output_folder`.

        Notes
        -----
        - The image is resized to match the dimensions of the template image.
        - Gaussian blurring is applied to the form and template images.
            - The kernel size is hard-coded to (5, 5) for both images.
        - Canny edge detection is used to detect edges in both images.
            - Edge lower and upper intensity thresholds are hard-coded to 50
            and 150.
        - Edges are detected and text within bounding boxes are extracted.
        """
        form = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if form is None:
            raise FileNotFoundError(f"Form image not found at {image_path}")

        form = cv2.resize(form, (template.shape[1], template.shape[0]))

        blurred_template = cv2.GaussianBlur(template, (5, 5), 0)
        blurred_form = cv2.GaussianBlur(form, (5, 5), 0)

        edges_template = cv2.Canny(blurred_template, 50, 150)
        edges_form = cv2.Canny(blurred_form, 50, 150)

        contours_template, _ = cv2.findContours(
            edges_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_form, _ = cv2.findContours(
            edges_form, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        template_boxes = self.get_bounding_boxes(contours_template)
        form_boxes = self.get_bounding_boxes(contours_form)

        if not template_boxes or not form_boxes:
            print(f"No bounding boxes detected in {image_path}")
            return

        self.extract_regions_and_save(
            template_boxes, form_boxes, form, output_folder, cad_id
        )

    def process_ocr(self, image_path, output_folder, cad_id):
        """
        Process a single image to extract and clean text, and redact names.

        Parameters
        ----------
        image_path : str
            Path to the image file to process.
        output_folder : str
            Folder to store the extracted text and images.
        cad_id : str
            Identifier for the current file, used in naming output files.

        Returns
        -------
        str
            The cleaned and processed text extracted from the image.

        Notes
        -----
        - Text extraction is performed using Tesseract OCR.
        - Remove specific unwanted text patterns, date/time stamps, and
        special characters.
        - Collapse multiple spaces into a single space.
        - Convert text to uppercase.
        - Personal names are anonymized in two passes:
        1. Detected names using SpaCy's Named Entity Recognition (NER) are replaced with "REDACTED."
        2. Known names from `common_names.csv` are identified and replaced with "REDACTED."
        - The processed text is returned as a single string.
        """

        form = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if form is None:
            raise FileNotFoundError(f"Form image not found at {image_path}")

        text = pytesseract.image_to_string(form)

        # Clean up the text by removing specific strings

        # Remove unwanted licensing text (case-insensitive) through end of doc
        text = re.sub(
            r"THIS DOCUMENT WAS CREATED BY AN APPLICATION THAT ISN’T LICENSED TO USE NOVAPDF.*",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove unwanted redaction log starting from "REDACTION LOG"
        # through the end of the document
        text = re.sub(
            r"REDACTION LOG.*",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove lines that start with "REDACTION DATE: " (case-insensitive)
        text = re.sub(r"^REDACTION DATE:.*\n?", "", text, flags=re.IGNORECASE)

        # Remove all date and time stamps (e.g., "1/4/2020 17:44:59")
        text = re.sub(
            r"\b\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}\b", "", text
        )

        # Remove all special characters, keep spaces
        text = re.sub(r"[^A-Za-z0-9\s]", "", text)

        # Collapse multiple spaces into a single space
        text = re.sub(r"\s{2,}", " ", text).strip()

        # Remove names of individuals
        # First pass: Use SpaCy NER and replace with "PERSON"
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                text = text.replace(ent.text, "REDACTED")

        # Second pass: Replace first names from common_names.csv
        for name in self.unique_names:
            # Use regex to match whole word only
            pattern = r"\b" + re.escape(name) + r"\b"

            # Check if the name exists in the text using re.search
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, "REDACTED", text, flags=re.IGNORECASE)

        # Convert all text to upper case.
        text = re.sub(r"\s+", " ", text).strip().upper()

        return text

    def get_bounding_boxes(
        self, contours, min_area=100, min_width=100, min_height=50
    ):
        """
        Method to compute bounding boxes for contours.
        Filter based on area, width, and height.

        Parameters
        ----------
        contours : list
            List of contours to process
        min_area : int, optional
            Minimum area of a contour to be considered a bounding box.
        min_width : int, optional
            Minimum width of a contour to be considered part of a bounding box.
        min_height : int, optional
            Minimum height of a contour to be considered part of a bounding box.

        Returns
        -------
        list of tuple
            A list of bounding boxes, where each bounding box is represented as
            a tuple (x1, y1, x2, y2), with:
            - (x1, y1): Coordinates of the top-left corner of the bounding box.
            - (x2, y2): Coordinates of the bottom-right corner of the box.

        Notes
        -----
        - Bounding boxes are computed using OpenCV's `cv2.boundingRect` function
        - Only contours with an area greater than `min_area` are considered.
        - Bounding boxes must also exceed both `min_width` and `min_height`
        """
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                if w > min_width and h > min_height:
                    boxes.append((x, y, x + w, y + h))
        return boxes

    def extract_regions_and_save(
        self, template_boxes, form_boxes, form, output_folder, cad_id
    ):
        """
        Extract regions from the form image based on template bounding boxes
        and save extracted text.

        Parameters
        ----------
        template_boxes : list of tuple
            List of bounding boxes represented as (x1, y1, x2, y2).
        form_boxes : list of tuple
            List of bounding boxes represented as (x1, y1, x2, y2).
        form : numpy.ndarray
            Grayscale image of the form from which regions will be extracted.
        output_folder : str
            Path to the folder where extracted images and text will be saved.
        cad_id : str
            Identifier for the current form file, used for naming output files.

        Returns
        -------
        None
            Saves extracted region images and text files to the output folder.

        Notes
        -----
        - The regions of `form_boxes` are matched with bounding boxes in
        `template_boxes` based on proximity (within a tolerance of 100 pixels).
        - Extracted regions are saved as PNG images in the `boxes`
        subdirectory of `output_folder`.
        - Text is extracted from each region using Tesseract OCR, cleaned, and
        saved to a narrative text file.
        - If found, the narrative section is identified by the keyword "NARRATIVE"

        Side Effects
        ------------
        - Creates directories for saving extracted images and narrative text
        - Does not handle duplicates well. If multiple boxes are close to each
        other, the output will be posted to a separate directory with (n) in
        the name.
        """
        box_count = 0
        officer_narrative = "UNKNOWN EVENTS"
        output_boxes_folder = os.path.join(output_folder, "boxes")
        os.makedirs(output_boxes_folder, exist_ok=True)

        for x1, y1, x2, y2 in template_boxes:
            for fx1, fy1, fx2, fy2 in form_boxes:
                if abs(x1 - fx1) < 100 and abs(y1 - fy1) < 100:
                    box_image = form[fy1:fy2, fx1:fx2]
                    box_image_path = os.path.join(
                        output_boxes_folder, f"box_{box_count}.png"
                    )
                    cv2.imwrite(box_image_path, box_image)
                    text = pytesseract.image_to_string(box_image)
                    text = re.sub(r"\s+", " ", text).strip().upper()

                    # Extract narrative and severity information from the text
                    if "NARRATIVE" in text[:15]:
                        officer_narrative = text[10:]
                    # Excluding this for now
                    # if "SEVERITY" in text[:15]:
                    #     severity = text[15:16]
                    #     severity_desc = self.interpret_severity(severity)

                    box_count += 1

        # Save the narrative text to a file
        oh1_narrative = officer_narrative
        narrative_file_path = os.path.join(
            output_folder, f"oh1_narrative_{cad_id}.txt"
        )

        with open(narrative_file_path, "w") as f:
            f.write(oh1_narrative)

    def interpret_severity(self, severity_code):
        """
        Interpret the severity code from the extracted text.

        Parameters
        ----------
        severity_code : str
            Severity code extracted from the OH1 text.

        Returns
        -------
        str
            A string description corresponding to the severity code.

        Notes
        -----
        - If the `severity_code` is not found in the mapping, the default value
        returned is "UNKNOWN INJURY CAUSED BY".
        - This function assumes the severity codes follow a numerical format.
        """

        severity_map = {
            "1": "FATAL INJURY CAUSED BY ",
            "2": "SERIOUS INJURY CAUSED BY ",
            "3": "MINOR INJURY CAUSED BY ",
            "4": "INJURY POSSIBLE CAUSED BY ",
            "5": "PROPERTY DAMAGE CAUSED BY ",
        }
        return severity_map.get(severity_code, "UNKNOWN INJURY CAUSED BY ")


class PreprocessGCAT:
    """
    A class to Preprocess preprocess the data for the GCAT model

    This class handles the preprocessing of text data, including tokenization,
    stemming, stopword removal, and TF-IDF vectorization. It also splits the
    dataset into training and testing sets for model training and evaluation.

        Parameters:
            text_column(str): Name of the column containing the text data
            label_column(str) = Name of the column containing the label data
            test_size(float) = Ratio of the test set
            train_size(float) = 1 - test_size (Ratio of the training set)
            norm(str) = Normalization method
            vocabulary(np.array) = Predefined vocabulary for the TF-IDF vectorizer
            min_df(float) = Minimum document frequency ratio
            max_df(float) = Maximum document frequency ratio
            max_features(float) = Maximum number of features returned from TF-IDF vectorizer


    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the text and label columns.
    text_column : str
        Name of the column containing the text data.
    label_column : str
        Name of the column containing the label data.
    test_size : float, optional, default=0.2
        Ratio of the test set size to the total dataset size.
    norm : str, optional, default='l2'
        Normalization method used by the TF-IDF vectorizer.
    vocabulary : np.array, optional, default=None
        Predefined vocabulary for the TF-IDF vectorizer. If None, the
        vocabulary will be determined from the input data.
    min_df : float, optional, default=0.05
        Minimum document frequency ratio for terms to be included in the TF-IDF matrix.
    max_df : float, optional, default=0.9
        Maximum document frequency ratio for terms to be included in the TF-IDF matrix.
    max_features : int, optional, default=500
        Maximum number of features to retain in the TF-IDF matrix.

    Attributes
    ----------
    ps : nltk.stem.PorterStemmer
        Stemmer used for reducing words to their root form.
    stopWords : set
        Set of stopwords from the NLTK library.
    charfilter : re.Pattern
        Regular expression pattern to filter characters (alphabets only).
    vec : sklearn.feature_extraction.text.TfidfVectorizer
        TF-IDF vectorizer used for feature extraction from text data.
    X_train : pandas.Series
        Training set containing text data.
    X_test : pandas.Series
        Testing set containing text data.
    y_train : pandas.Series
        Training set containing labels.
    y_test : pandas.Series
        Testing set containing labels.

    Returns
    -------
    None
        Prints the number of rows in the training and testing sets upon init.

    Notes and AI Highlights
    -----
    - The dataset is split into training and testing sets using a fixed random seed.
    - The training size is calculated as `1 - test_size`.
    - The TF-IDF vectorizer applies tokenization, normalization, and
    dimensionality reduction based on the provided or generated vocabulary.        
    """

    def __init__(
        self,
        df,
        text_column,
        label_column,
        test_size=0.2,
        norm="l2",
        vocabulary=None,
        min_df=0.05,
        max_df=0.9,
        max_features=500,
    ):
        """
        Initialize the PreprocessGCAT class.

        See the class-level docstring for a description of parameters.
        """

        self.df = df
        self.text_column = text_column
        self.label_column = label_column
        self.test_size = test_size
        self.train_size = 1 - test_size
        self.norm = norm
        self.vocabulary = vocabulary
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features

        self.ps = PorterStemmer()
        self.stopWords = set(nltk.corpus.stopwords.words("english"))
        self.charfilter = re.compile("[a-zA-Z]+")
        self.vec = TfidfVectorizer(
            tokenizer=self.CCR_Tokenizer,
            norm=self.norm,
            vocabulary=self.vocabulary,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
        )

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                self.df[self.text_column],
                self.df[self.label_column],
                train_size=self.train_size,
                random_state=123,
            )
        )

        print("Dataset has been split into training and testing sets.")

        print(f"Number of rows in total: {len(self.df)}")
        print(f"Number of rows in X_train: {len(self.X_train)}")
        print(f"Number of rows in X_test: {len(self.X_test)}")

    def CCR_Tokenizer(self, text):
        """
        Stem and tokenize raw text data, removing stopwords and
        non-alphabetic tokens.

        Parameters
        ----------
        text : str
            Raw text data to be tokenized and processed.

        Returns
        -------
        list of str
            A list of processed tokens after stemming, stopword removal, and
            filtering non-alphabetic tokens.
        """
        words = map(lambda word: word.lower(), nltk.word_tokenize(text))
        words = [word for word in words if word not in self.stopWords]
        tokens = list(map(lambda token: self.ps.stem(token), words))
        ntokens = list(
            filter(lambda token: self.charfilter.match(token), tokens)
        )
        return ntokens

    def fit_and_evaluate_tfidf_vector(self):
        """
        Fit and evaluate the TF-IDF vectorizer.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Prints summary statistics of the fitted TF-IDF vectorizer.

        Notes
        -----
        - The function fits the TF-IDF vectorizer to the training data
        (`X_train`) and transforms the text data into a sparse matrix.
        - It converts the sparse matrix to a dense format and computes the
        following summary statistics:
            - Number of features/terms in the vector.
            - Minimum value of the TF-IDF matrix.
            - Maximum value of the TF-IDF matrix.
            - Percentiles (25th, 50th, 75th, and 95th) from the TF-IDF matrix.
        """
        X_tfidf = self.vec.fit_transform(self.X_train)
        X_tfidf_dense = X_tfidf.todense()
        X_tfidf_array = np.array(X_tfidf_dense)
        vector_size = X_tfidf.shape[1]
        print("Number of Features/Terms in vector):", vector_size)
        print("Min value:", np.min(X_tfidf_array))
        print("Max value:", np.max(X_tfidf_array))
        print("Percentiles:", np.percentile(X_tfidf_array, [25, 50, 75, 95]))

    def create_doc_term_matrix(self):
        """
        Create the document-term matrix (DTM).

        Parameters
        ----------
        None

        Returns
        -------
        dtm : scipy.sparse._csr.csr_matrix
            A sparse matrix representation of the document-term matrix.

        Notes
        -----
        - This function uses `self.X_train`, which is expected as a pandas
        Series or a list of text documents representing the training dataset.
        - The TF-IDF vectorizer (`self.vec`) is used to transform the documents
        into the matrix representation.
        - The matrix represents the frequency or importance of terms within the
        training dataset (`X_train`).
        - Each row corresponds to a document and each col to a term or feature.
        """
        docs = list(self.X_train)
        dtm = self.vec.fit_transform(docs)
        return dtm

    def pca_analysis(self, dtm):
        """
        Principal Component Analysis (PCA) on the document-term matrix (DTM) to
        determine the number of components required to explain 95% of the
        variance, and visualize the results.

        Parameters
        ----------
        dtm : scipy.sparse._csr.csr_matrix
            The document-term matrix (DTM) to be analyzed.

        Returns
        -------
        explained_var : list of float
            List of cumulative explained variance ratios for diff nbr of components.
        components : int
            The min nbr of principal components req to explain at least 95% of the variance.

        Visualizations
        --------------
        1. A plot showing the proportion of explained variance against the nbr
        of PCA components, with reference lines indicating 95% variance and the
        optimal number of components.
        2. A scatterplot of the first two PCA dimensions, with points colored
        based on class labels.
        """
        # Perform PCA on the DTM
        pca_temp = PCA().fit(dtm.toarray())

        # Calculate cumulative variance and components
        cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        components = np.argmax(cumulative_variance >= 0.95) + 1
        components_range = components + (components % 5) + 5
        explained_var = []
        for comp in range(1, components_range, 5):
            pca = PCA(n_components=comp)
            pca.fit(dtm.toarray())
            explained_var.append(pca.explained_variance_ratio_.sum())

        print("Explained Variance: {:.2f}%".format(explained_var[-1] * 100))
        print("Number of Components: ", components)

        # Plot explained variance by number of components
        plt.figure()
        plt.plot(range(1, components_range, 5), explained_var, "ro")
        plt.axhline(y=0.95, color="b", linestyle="--", label="95% Variance")
        plt.axvline(
            x=components, color="g", linestyle="--", label="Num Components"
        )
        plt.xlabel("Number of Components")
        plt.ylabel("Proportion of Explained Variance")

        # Scattertlot PCA
        plt.figure()
        palette = np.array(sns.color_palette("hls", 10))
        pca = PCA(n_components=components)
        pca.fit(dtm.toarray())
        pca_dtm = pca.transform(dtm.toarray())
        explained_variance = pca.explained_variance_ratio_.sum()
        plt.scatter(
            pca_dtm[:, 0], pca_dtm[:, 1], c=palette[self.y_train.astype(int)]
        )
        plt.xlabel("Can1")
        plt.ylabel("Can2")
        print("Explained Variance: {:.2f}%".format(explained_variance * 100))

        return explained_var, components

    def evaluate_balancers(self, models, class_balancers, n_components=40):
        """
        Procedure to quickly compare different class imbalance correction
        methods before performing grid search. Intended to be used for
        exploration of class imbalance correction.

        Parameters:
            models(list): List of models to evaluate
            class_balancers(list): List of class imbalance correction methods
            n_components(int): Number of components for PCA

        Returns:
            balancer_eval(pd.DataFrame): DataFrame of evaluation
        """

        """
        Evaluate different class imbalance correction methods and compare their
        performance across multiple models. Exploratory analysis of class
        imbalance with a grid search before performing correction techniques.

        Parameters
        ----------
        models : list of tuples
            A list of model tuples where each tuple contains the
            model name (str) and the corresponding model object
            (e.g., `("Logistic Regression", LogisticRegression())`).
        class_balancers : list
            A list of class imbalance correction methods.
        n_components : int, optional
            The desired number of Principal Component Analysis (PCA) components
            to retain in the pipeline.

        Returns
        -------
        balancer_eval : pandas.DataFrame
            A DataFrame with evaluation metrics for each combination of model
            and class imbalance correction method.
                - "Model": The name of the model.
                - "Balancer": The name of the class imbalance correction method.
                - "Confusion Matrix": The confusion matrix of the predictions.
                - "Macro Avg Precision": The macro-averaged precision score.
                - "Macro Avg Recall": The macro-averaged recall score.
                - "Macro Avg F1": The macro-averaged F1 score.

        Notes
        -----
        - For each combination of model and class balancer, the method builds a
        pipeline that includes the vectorizer (`self.vec`), the class balancer,
        PCA transformation, and the model.
         - The method calculates confusion matrices and macro-averaged
        classification metrics, storing them in a DataFrame for comparison.
        """
        models = models
        class_balancers = class_balancers
        n_components = n_components

        balancer_eval = pd.DataFrame(
            columns=[
                "Model",
                "Balancer",
                "Confusion Matrix",
                "Macro Avg Precision",
                "Macro Avg Recall",
                "Macro Avg F1",
            ]
        )

        # Run through all combinations of models and balancers
        for balancer in class_balancers:
            for m_name, model in models:
                pipeline = Pipeline(
                    [
                        ("vec", self.vec),
                        ("class_balancer", balancer),
                        ("pca", PCA(n_components=n_components)),
                        (m_name, model),
                    ]
                )

                # Fit and predict
                pipeline.fit(self.X_train, self.y_train)
                y_pred = pipeline.predict(self.X_test)

                # Confusion matrix and classification report
                conf_matrix = confusion_matrix(self.y_test, y_pred)
                report = classification_report(
                    self.y_test, y_pred, output_dict=True
                )["macro avg"]

                # Append results to the DataFrame
                new_row = pd.DataFrame(
                    {
                        "Model": [m_name],
                        "Balancer": [str(balancer)],
                        "Confusion Matrix": [conf_matrix],
                        "Macro Avg Precision": [report["precision"]],
                        "Macro Avg Recall": [report["recall"]],
                        "Macro Avg F1": [report["f1-score"]],
                    }
                )
                new_row_rmv_na = new_row.dropna(axis=1, how="all")
                balancer_eval = pd.concat(
                    [balancer_eval, new_row_rmv_na], ignore_index=True
                )

        return balancer_eval


class GenBikeCleNarrative:
    """
    A class to generate BikeCLE narrative using Google AI services.

    Attributes
    ----------
    google_api_key : str or None
        The Google API key for accessing Google services. Defaults to None.

    AI Highlights
    -------------
    - NLP text summarization using the Google Gemini model.
    - Fine-tuning of Gemini LLM with BikeCLE narrative data using low-code
    Google AI services.
    """

    def __init__(
        self,
        google_api_key="CCR_API",
    ):

        self.google_api_key = google_api_key
        genai.configure(api_key=os.environ[self.google_api_key])

        """
        Instantiates Google API as a class attribute for accessing Google services.

        Parameters
        ----------
        google_api_key : str or None
            The Google API key.

        Returns
        -------
        None
        """

    def summarize(self, concat_text=None, max_retries=5):
        """
        Summarize the provided text using the Googles GenAI model that has been
        fine-tuned on BikeCLE narrative data.

        Parameters
        ----------
        concat_text : str
            Concatenation of CAD and OH1 text containing dets of the collision.
        max_retries : int, optional
            The maximum number of retry attempts in case of a failure due to
            resource exhaustion. Default is 5.

        Returns
        -------
        str
            The summarized text generated by the model.

        Raises
        ------
        Exception
            If the maximum number of retries is reached without successfully summarizing the text.
        """
        model = genai.GenerativeModel(
            "tunedModels/bikecleinputdf-4pk28onmojpn"
        )  # Fine-tuned Gemini model for bikecle input df
        #

        retries = 0

        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    [concat_text],
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
                return response.text

            except ResourceExhausted as e:
                retries += 1
                print(
                    f"Resource exhausted. Attempt {attempt + 1}/{retries}. "
                    f"Retrying in 15 seconds...\nError: {e}"
                )
                time.sleep(15)
        raise Exception("Max retries reached. Unable to summarize.")
