# Map src directory
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "../."))
src_dir = os.path.join(root_dir,"src")
sys.path.append(src_dir)
import complete_collision
folder_path = os.path.join(root_dir, 'data', 'raw', 'oh1_pdf_files')
output_base_folder = os.path.join(root_dir, 'data', 'processed', 'oh1_images')
template_path = os.path.join(root_dir, 'docs', 'oh1_page_1_template.png')
print(f"Starting extraction from {folder_path} and outputing to {output_base_folder} using {template_path} as a format")
pdf_extractor = complete_collision.PdfTextExtraction()
pdf_extractor.process_oh1_pdfs_in_folder(
    folder_path, output_base_folder, template_path
)