# Complete Collision Recorder

Roughly half of all traffic collisions are not documented in official crash
reports. This issue has been understood and researched for more than 50 years
but it is still recommended practice to base traffic engineering decision on
official data only. This project attempts to use AI to extract information from
alternative unstructured data sources, such as police dispatch reports, and
operationalize information from those reports into a usable structured format
and to create natural language descriptions of the collisions.

## Project Description
A two-step AI process using an NLP categorization task to extract:

- Driver vehicle type
- Driver maneuver
- Pedestrian characteristic
- Pedestrian vehicle type
- Area of collision

And to use raw text and those defined crash characteristics as inputs into a
text generator to create natural language descriptions of collisions.

## Project Directory Structure

```plaintext
complete_collision_recorder/
│
├── data/                   # Directory for storing data files
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data ready for analysis
│
├── notebooks/              # Jupyter notebooks for analysis
│   └── dev_20240909.ipynb  # Main analysis notebook
│
├── src/                    # Source code (Python scripts and modules)
│   ├── __init__.py         # Marks this folder as a package
│   ├── utils.py            # Utility functions for general use
│   └── complete_collision.py # Project-specific logic and functions
│
├── tests/                  # Unit tests for the project
│   ├── test_utils.py       # Tests for utility functions
│   └── test_complete_collision.py # Tests for project-specific functions
│
├── README.md               # Project overview and documentation
├── requirements.txt        # Python dependencies
└── setup.py                # Installation script
