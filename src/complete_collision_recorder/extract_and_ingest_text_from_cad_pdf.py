# Standard Python Libraries
import os

# Local Modules
import complete_collision as cc
import utils as u


# Step 1: Instantiate the PdfTextExtraction object
pdf_extractor = cc.PdfTextExtraction()

# Step 2: Map the project directories
root_dir, src_dir, data_dir, models_dir = u.map_project_directories()

# Step 3: Assign the output folder for the SSA statistical names
output_ssa_folder = os.path.join(root_dir, "data", "lookup", "ssa_names")
print(
    f"Starting extraction from {pdf_extractor.ssa_url} and outputing to {output_ssa_folder} "
)

# Step 4: Create the SSA statistical name dataset
pdf_extractor.create_common_name_dataset(output_ssa_folder)

# Step 5: Assign the input folder for the CAD PDF files and the output folder
# for the CAD images and extracted text
folder_path = os.path.join(root_dir, "data", "raw", "cad_pdf_files")
output_base_folder = os.path.join(root_dir, "data", "processed", "cad_images")

# Step 6: Extract the text from the CAD PDF files
print(
    f"Starting extraction from {folder_path} and outputing to {output_base_folder} "
)
pdf_extractor.process_cad_pdfs_in_folder(folder_path, output_base_folder)
