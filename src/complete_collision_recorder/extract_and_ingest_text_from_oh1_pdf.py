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

# Step 5: Assign the input folder for the OH1 PDF files and the output folder
# for the OH1 page images and extracted text
folder_path = os.path.join(root_dir, "data", "raw", "oh1_pdf_files")
output_base_folder = os.path.join(root_dir, "data", "processed", "oh1_images")

# Step 5.1: Assign the template path for the OH1 page
template_path = os.path.join(root_dir, "docs", "oh1_page_1_template.png")

# Step 6: Extract the text from the OH1 PDF files
print(
    f"Starting extraction from {folder_path} and outputing to {output_base_folder} using {template_path} as a format"
)
pdf_extractor.process_oh1_pdfs_in_folder(
    folder_path, output_base_folder, template_path
)
