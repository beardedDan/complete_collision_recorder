# General Packages
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time

# OCR Packages
import cv2
from pdf2image import convert_from_path
import pytesseract

# Preprocessing Packages
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Model Selection Packages
from sklearn.model_selection import (
    train_test_split,
)
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Google Gemini GenAI Packages
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# Warning Suppression
import warnings

warnings.filterwarnings(
    "ignore",
    message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None",
)

# Load environmental variables
load_dotenv()


class PdfTextExtraction:
    """
    Corresponds to Functional Requirements 01, 02, and 03 in System Design.
    See details in the README.md or in the System Design document:
    /docs/system_design_and_functional_nonfunctional_requirements.md
    """

    def __init__(self):

        root_directory = os.path.abspath(os.path.join(os.getcwd(), "../."))
        log_directory = os.path.join(root_directory, "logs")
        os.makedirs(log_directory, exist_ok=True)
        log_file_path = os.path.join(log_directory, "pdf_text_extraction.log")

        # Reference: https://realpython.com/python-logging/
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

    def process_cad_pdfs_in_folder(
        self, folder_path, output_base_folder, dpi=300
    ):
        """
        Process all pages of CAD PDF files
        convert to image,
        extract text with OCR,
        save results.

        Args:
            folder_path (str): Folder with saved PDFs.
            output_base_folder (str): Folder to save processed files.
            dpi (int): Dots Per Inch for image conversion default 300

        Returns:
            None. Saves output files to subfolders within output_base_folder
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

                        with open(output_text_file, "a") as text_file:
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
        Process the first page of all PDF files
        convert to image,
        matching forms with a template,
        detect bounding boxes,
        extract text,
        save results.

        Args:
            folder_path (str): Folder with saved PDFs.
            output_base_folder (str): Folder to save processed files.
            template_path (str): File path of page 1 template.
            dpi (int): Dots Per Inch for image conversion default 300

        Returns:
            None. Saves output files to subfolders within output_base_folder
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
        Convert all pages of the PDF to an image.

        Args:
            pdf_path (str): Path to the PDF file.
            output_folder (str): Folder to store the resulting image.
            dpi (int): DPI for image conversion.

        Returns:
            str: A list of file paths of all saved images.
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
        Convert only the first page of the PDF to an image.

        Args:
            pdf_path (str): Path to the PDF file.
            output_folder (str): Folder to store the resulting image.
            dpi (int): DPI for image conversion.

        Returns:
            str: File path of the saved first page image.
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
        Process a single image:
        resize,
        blur,
        detect edges,
        find contours,
        match bounding boxes within tolerance,
        extract text,
        interpret severity.

        Args:
            image_path (str): Path to the image file to process.
            template (np.array): Template image for comparison.
            output_folder (str): Folder to store the extracted images and text.
            cad_id (str): Identifier for the current file (used in naming output).

        Returns:
            None. Saves files to output_folder
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
        Process a single image:
        extract text,
        interpret severity.

        Args:
            image_path (str): Path to the image file to process.
            output_folder (str): Folder to store the extracted images and text.
            cad_id (str): Identifier for the current file (used in naming output).

        Returns:
            None. Saves files to output_folder
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

        # Remove unwanted redaction log (case-insensitive)
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

        # Convert all text to upper case.
        text = re.sub(r"\s+", " ", text).strip().upper()

        return text

    def get_bounding_boxes(
        self, contours, min_area=100, min_width=100, min_height=50
    ):
        """
        Get bounding boxes for the contours

        Args:
            contours (list): List of contours to process.
            min_area (int): Minimum area of a contour to be considered.
            min_width (int): Minimum width of a bounding box.
            min_height (int): Minimum height of a bounding box.

        Returns:
            List[tuple]: List of bounding boxes as (x1, y1, x2, y2).
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
        Extract the regions from the form based on the template
        save the extracted text.

        Args:
            template_boxes (list): List of bounding boxes from the template.
            form_boxes (list): List of bounding boxes from the form.
            form (np.array): Form image to extract regions from.
            output_folder (str): Folder to save the extracted images and text.
            cad_id (str): Identifier for the current file.

        Returns:
            None
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

        Args:
            severity_code (str): Severity code (1-5) extracted from the text.

        Returns:
            str: A string description of the severity.
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

    # PreprocessGCAT class is used to preprocess the data for the GCAT model

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
        Initiatize the class.
        Define class variables
        Define stemmer, stopwords, and character filter
        Define TF-IDF vectorizer
        Split the dataset into training and testing sets

        Args:
            text_column(str): Name of the column containing the text data
            label_column(str) = Name of the column containing the label data
            test_size(float) = Ratio of the test set
            train_size(float) = 1 - test_size (Ratio of the training set)
            norm(str) = Normalization method
            vocabulary(np.array) = Predefined vocabulary for the TF-IDF vectorizer
            min_df(float) = Minimum document frequency ratio
            max_df(float) = Maximum document frequency ratio
            max_features(float) = Maximum number of features returned from TF-IDF vectorizer

        Returns:
            Prints the number of rows in the training and testing sets
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
        Stem and tokenize raw text data.

        Args:
            text(str): Raw text data to be tokenized

        Returns:
            ntokens: List of tokens produced with tokenizer
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
        Procedure to fit and evaluate the TF-IDF vectorizer

        Args:
            None

        Returns:
            Prints summary statistics of the TF-IDF vectorizer
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
        Create the document term matrix

        Args:
            None

        Returns:
            dtm(scipy.sparse._csr.csr_matrix): Document term matrix
        """

        docs = list(self.X_train)
        dtm = self.vec.fit_transform(docs)
        return dtm

    def pca_analysis(self, dtm):
        """
        Procedure to analyze the PCA of the document term matrix, plot
        explained variance, return the ideal number of components.

        Args:
            dtm(scipy.sparse._csr.csr_matrix): Document term matrix

        Returns:
            explained_var(list): List of explained variance ratios
            components(int): Number of components to explain 95% variance
        """

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

        Args:
            models(list): List of models to evaluate
            class_balancers(list): List of class imbalance correction methods
            n_components(int): Number of components for PCA

        Returns:
            balancer_eval(pd.DataFrame): DataFrame of evaluation
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

    def __init__(
        self,
        google_api_key=None,
    ):

        self.google_api_key = google_api_key
        genai.configure(api_key=os.environ[self.google_api_key])

        """
        Initiatize the class.
        Define Google API Key

        Args:
            google_api_key(str): Google API Key

        Returns:
            None
        """

    def summarize(self, concat_text=None, max_retries=5):

        # concat_text = concat_text
        # max_retries = max_retries
        """
        Summarize the text using the GenAI model

        Args:
            concat_text(str): Concatenated text of all detail of collision

        Returns:
            response.text(str): Summarized text
        """

        model = genai.GenerativeModel(
            "tunedModels/bikecleinputdf-4pk28onmojpn"
        )  # Fine-tuned Gemini model for bikecle input df

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
