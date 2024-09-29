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
│   ├── raw/                # Git ignored - Raw data files
│   └── processed/          # Git ignored - Processed data ready for analysis
│
├── notebooks/              # Jupyter notebooks for analysis
│   └── dev_20240909.ipynb  # Initial dev notebook
│
├── src/                    # Source code (Python scripts and modules)
│   ├── __init__.py         # Marks this folder as a package
│   ├── main.py             # CLI entry point
│   ├── utils.py            # Utility functions for general use
│   └── complete_collision.py # Project-specific logic and functions
│
├── tests/                  # Unit tests for the project
│   ├── test_utils.py       # Tests for utility functions
│   └── test_complete_collision.py # Tests for project-specific functions
|
├── docs/                   # Project specific documentation
│
├── README.md               # Project overview and documentation
├── LICENSE.txt             # Licensing and usage permissions
└── pyproject.toml          # Python project build info