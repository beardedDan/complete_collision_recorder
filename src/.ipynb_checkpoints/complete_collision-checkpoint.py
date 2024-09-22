import cv2
import os
import re
from pdf2image import convert_from_path
import pytesseract


def process_pdfs_in_folder(
    folder_path, output_base_folder, template_path, dpi=300
):
    """
    Process the first page of all PDF files
    convert to image,
    matching forms with a template,
    detect bounding boxes,
    extract text,
    save results.

    Args:
        folder_path (str): Folder with saved PDFs.
        output_base_folder (str): Folder to save processed files.
        template_path (str): File path of page 1 template.
        dpi (int): Dots Per Inch for image conversion default 300

    Returns:
        None
    """

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Template image not found {template_path}")

    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith(".pdf"):
            cad_id = os.path.splitext(pdf_file)[0]
            pdf_path = os.path.join(folder_path, pdf_file)
            output_folder = os.path.join(output_base_folder, cad_id)
            os.makedirs(output_folder, exist_ok=True)

            # Step 1: Convert only the first page of the PDF to an image
            image_file = convert_first_page_to_image(
                pdf_path, output_folder, dpi
            )
            if image_file:
                process_image(image_file, template, output_folder, cad_id)
            print("Processed PDF: ", pdf_file)


def convert_first_page_to_image(pdf_path, output_folder, dpi=300):
    """
    Convert only the first page of the PDF to an image.

    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Folder to store the resulting image.
        dpi (int): DPI for image conversion.

    Returns:
        str: File path of the saved first page image.
    """
    images = convert_from_path(
        pdf_path, dpi=dpi, first_page=1, last_page=1
    )  # Convert first page only
    if images:
        image_path = os.path.join(output_folder, "page_1.png")
        images[0].save(image_path, "PNG")
        return image_path
    return None


def process_image(image_path, template, output_folder, cad_id):
    """
    Process a single image:
    resize,
    blur,
    detect edges,
    find contours,
    match bounding boxes within tolerance,
    extract text,
    interpret severity.

    Args:
        image_path (str): Path to the image file to process.
        template (np.array): Template image for comparison.
        output_folder (str): Folder to store the extracted images and text.
        cad_id (str): Identifier for the current file (used in naming output).

    Returns:
        None
    """
    form = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if form is None:
        raise FileNotFoundError(f"Form image not found at {image_path}")

    form = cv2.resize(form, (template.shape[1], template.shape[0]))

    blurred_template = cv2.GaussianBlur(template, (5, 5), 0)
    blurred_form = cv2.GaussianBlur(form, (5, 5), 0)

    edges_template = cv2.Canny(blurred_template, 50, 150)
    edges_form = cv2.Canny(blurred_form, 50, 150)

    contours_template, _ = cv2.findContours(
        edges_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_form, _ = cv2.findContours(
        edges_form, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    template_boxes = get_bounding_boxes(contours_template)
    form_boxes = get_bounding_boxes(contours_form)

    if not template_boxes or not form_boxes:
        print(f"No bounding boxes detected in {image_path}")
        return

    extract_regions_and_save(
        template_boxes, form_boxes, form, output_folder, cad_id
    )


def get_bounding_boxes(contours, min_area=100, min_width=100, min_height=50):
    """
    Get bounding boxes for the contours

    Args:
        contours (list): List of contours to process.
        min_area (int): Minimum area of a contour to be considered.
        min_width (int): Minimum width of a bounding box.
        min_height (int): Minimum height of a bounding box.

    Returns:
        List[tuple]: List of bounding boxes as (x1, y1, x2, y2).
    """
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            if w > min_width and h > min_height:
                boxes.append((x, y, x + w, y + h))
    return boxes


def extract_regions_and_save(
    template_boxes, form_boxes, form, output_folder, cad_id
):
    """
    Extract the regions from the form based on the template
    save the extracted text.

    Args:
        template_boxes (list): List of bounding boxes from the template.
        form_boxes (list): List of bounding boxes from the form.
        form (np.array): Form image to extract regions from.
        output_folder (str): Folder to save the extracted images and text.
        cad_id (str): Identifier for the current file.

    Returns:
        None
    """
    box_count = 0
    officer_narrative = "UNKNOWN EVENTS"
    severity_desc = "UNKNOWN INJURY CAUSED BY "
    output_boxes_folder = os.path.join(output_folder, "boxes")
    os.makedirs(output_boxes_folder, exist_ok=True)

    for x1, y1, x2, y2 in template_boxes:
        for fx1, fy1, fx2, fy2 in form_boxes:
            if abs(x1 - fx1) < 100 and abs(y1 - fy1) < 100:
                box_image = form[fy1:fy2, fx1:fx2]
                box_image_path = os.path.join(
                    output_boxes_folder, f"box_{box_count}.png"
                )
                cv2.imwrite(box_image_path, box_image)
                text = pytesseract.image_to_string(box_image)
                text = re.sub(r"\s+", " ", text).strip().upper()

                # Extract narrative and severity information from the text
                if "NARRATIVE" in text[:15]:
                    officer_narrative = text[10:]
                if "SEVERITY" in text[:15]:
                    severity = text[15:16]
                    severity_desc = interpret_severity(severity)

                box_count += 1

    # Save the narrative text to a file
    oh1_narrative = severity_desc + officer_narrative
    narrative_file_path = os.path.join(
        output_folder, f"oh1_narrative_{cad_id}.txt"
    )

    with open(narrative_file_path, "w") as f:
        f.write(oh1_narrative)


def interpret_severity(severity_code):
    """
    Interpret the severity code from the extracted text.

    Args:
        severity_code (str): Severity code (1-5) extracted from the text.

    Returns:
        str: A string description of the severity.
    """
    severity_map = {
        "1": "FATAL INJURY CAUSED BY ",
        "2": "SERIOUS INJURY CAUSED BY ",
        "3": "MINOR INJURY CAUSED BY ",
        "4": "INJURY POSSIBLE CAUSED BY ",
        "5": "PROPERTY DAMAGE CAUSED BY ",
    }
    return severity_map.get(severity_code, "UNKNOWN INJURY CAUSED BY ")
