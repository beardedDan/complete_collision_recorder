# Import General Libraries
import os

# Import project scripts
import complete_collision as cc
import utils as u

# Map the project directories
root_dir, src_dir, data_dir, models_dir = u.map_project_directories()

folder_path = os.path.join(root_dir, 'data', 'raw', 'oh1_pdf_files')
output_base_folder = os.path.join(root_dir, 'data', 'processed', 'oh1_images')
template_path = os.path.join(root_dir, 'docs', 'oh1_page_1_template.png')
print(f"Starting extraction from {folder_path} and outputing to {output_base_folder} using {template_path} as a format")
pdf_extractor = cc.PdfTextExtraction()
pdf_extractor.process_oh1_pdfs_in_folder(
    folder_path, output_base_folder, template_path
)