# Import General Libraries
import os

# Import project scripts
import complete_collision as cc
import utils as u

# Map the project directories
root_dir, src_dir, data_dir, models_dir = u.map_project_directories()

# Assign the output folder for the SSA statistical names
output_ssa_folder = os.path.join(root_dir, 'data', 'lookup', 'ssa_names')
print(f"Starting extraction from {pdf_extractor.ssa_url} and outputing to {output_ssa_folder} ")

# Create the SSA statistical name dataset
pdf_extractor.create_common_name_dataset(
    output_ssa_folder
)
