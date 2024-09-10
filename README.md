# Complete Collision Recorder

## Project Description
A two-step AI process flow of NLP categorization predictive model to extract:

- Driver vehicle type
    - Unknown
    - Car
    - Large passenger vehicle
    - Very large vehicle
    - Motorcycle or Micro-mobility Device
- Driver maneuver
    - Unknown
    - Straight
    - Left Turn
    - Right Turn
    - Backward
    - Vehicle as weapon
- Pedestrian characteristic
    - Unknown
    - Adult
    - Child
    - Multiple
- Pedestrian vehicle type
    - Unknown
    - On foot
    - Bicycle
    - Micro-mobility Device
    - Wheelchair or stroller
- Area of collision
    - Unknown
    - Intersection
    - Mid-block
    - Bike lane
    - Driveway
    - Parking area


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
│   └── project_lib.py      # Project-specific logic and functions
│
├── tests/                  # Unit tests for the project
│   ├── test_utils.py       # Tests for utility functions
│   └── test_project_lib.py # Tests for project-specific functions
│
├── README.md               # Project overview and documentation
├── requirements.txt        # Python dependencies
└── setup.py                # Installation script
