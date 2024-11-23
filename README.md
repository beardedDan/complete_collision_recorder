# Complete Collision Recorder

Roughly half of all traffic collisions are not documented in official crash
reports. This is a problem because traffic engineers do not have a complete
safety profile of a roadway and are only able to consider fully documented 
collisions. This results in a serious survivorship bias and has been 
researched for more than 50 years but it is still recommended practice for 
traffic engineers to only use official data when performing safety studies. 
This project attempts to use AI to extract information from alternative 
unstructured data sources, such as police dispatch reports, and 
operationalize information from those reports into a usable structured 
format and to create natural language descriptions of the collisions.

## Project Description
Taking inputs from official crash reports and police reports, a two-step 
process using an NLP categorization task to extract:

- Driver vehicle type
- Driver maneuver
- Pedestrian characteristic
- Pedestrian vehicle type
- Area of collision

And then provide raw text and the extracted crash characteristics as inputs 
into a text generator to create natural language descriptions of collisions.

## System Design and Functional/Non-Functional Requirements

System pipeline and requirements are documented here: [System Design and Functional/Non-Functional Requirements](./docs/system_design_and_functional_nonfunctional_requirements.md
system_design_and functional_nonfunctional_requirements.md
)


## Project Directory Structure

```plaintext
complete_collision_recorder/
│
├── data/                   # Git ignored - Directory for storing data files
|   ├── lookup/             # Git ignored - Lookup files for SSA and GIS functions
│   ├── raw/                # Git ignored - Raw data files
│   └── processed/          # Git ignored - Processed data ready for analysis
│
├── notebooks/              # Jupyter notebooks for analysis
│   ├── analysis_compare_to_OSTATS.ipynb
│   ├── import_CAD_OH1_PDFs.ipynb.ipynb
│   ├── import_Combine_Index_Files.ipynb
│   ├── import_Download_File_IDs_in_Index.ipynb
│   ├── score_bikeCLE_Gemini.ipynb
│   ├── score_GCAT.ipynb
│   ├── train_and_evaluate_bikeCLE_Gemini.ipynb
│   └── train_GCAT_severity.ipynb
|
├── models/                 # Jupyter notebooks for analysis
│   ├── severity.pkl        # Pickle model for GCAT severity prediction
│   └── BikeCLE_narrative_fine_tune_training_results.png  #Screenshot of Gemini results
│
├── src/                    # Source code (Python scripts and modules)
│   ├── complete_collision_recorder  # Package folder
│   └── ├── __init__.py         # Marks this folder as a package
│       ├── __main__.py         # CLI entry point
|       ├── app.py              # Program to start webpage service
│       ├── utils.py            # Utility functions for general use
|       ├── assemble_text.py
|       ├── extract_and_ingest_text_from_cad_pdf.py
|       ├── extract_and_ingest_text_from_oh1_pdf.py
|       ├── score_and_describe_reports.py
│       └── complete_collision.py # Project-specific module
│
├── docs/                   # Project specific documentation
│
├── README.md               # Project overview and documentation
├── LICENSE.txt             # Licensing and usage permissions
└── pyproject.toml          # Python project build info