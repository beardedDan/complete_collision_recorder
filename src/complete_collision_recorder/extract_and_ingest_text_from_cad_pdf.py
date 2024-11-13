# Import General Libraries
import os

# Import project scripts
import complete_collision as cc
import utils as u

# Map the project directories
root_dir, src_dir, data_dir, models_dir = u.map_project_directories()

folder_path = os.path.join(root_dir, 'data', 'raw', 'cad_pdf_files')
output_base_folder = os.path.join(root_dir, 'data', 'processed', 'cad_images')
print(f"Starting extraction from {folder_path} and outputing to {output_base_folder} ")
pdf_extractor = cc.PdfTextExtraction()
pdf_extractor.process_cad_pdfs_in_folder(
    folder_path, output_base_folder
)