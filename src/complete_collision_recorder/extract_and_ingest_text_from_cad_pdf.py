# Map src directory
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "../."))
src_dir = os.path.join(root_dir,"src")
sys.path.append(src_dir)
import complete_collision
folder_path = os.path.join(root_dir, 'data', 'raw', 'cad_pdf_files')
output_base_folder = os.path.join(root_dir, 'data', 'processed', 'cad_images')
print(f"Starting extraction from {folder_path} and outputing to {output_base_folder} ")
pdf_extractor = complete_collision.PdfTextExtraction()
pdf_extractor.process_cad_pdfs_in_folder(
    folder_path, output_base_folder
)