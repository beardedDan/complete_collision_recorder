import subprocess
import os


def run_bin_program(script_name):
    script_path = os.path.join(os.getcwd(), "../bin", script_name)
    subprocess.run(["python", script_path], check=True)


def run_all():
    """
    The program will run these programs in order:

        1. extract_and_ingest_text_from_cad_pdf.py
        2. extract_and_ingest_text_from_oh1_pdf.py
        3. assemble_text.py
        4. score_and_describe_reports.py

    """
    print("Starting Collision Recorder CLI...")
    print("Running data extraction and prediction process...")

    run_bin_program("extract_and_ingest_text_from_cad_pdf.py")
    run_bin_program("extract_and_ingest_text_from_oh1_pdf.py")
    run_bin_program("assemble_text.py")
    print("Running score_and_describe_reports.py with --testing argument...")
    subprocess.run(
        ["python", "../bin/score_and_describe_reports.py", "--testing"],
        check=True,
    )

    print("All data extraction and prediction processes complete.")
    print("Stopping Collision Recorder CLI.")

def run_cad_import():
    """
    The program will run these programs in order:

        extract_and_ingest_text_from_cad_pdf.py

    """
    print("Extracting text from CAD files...")

    run_bin_program("extract_and_ingest_text_from_cad_pdf.py")

    print("Data extraction and prediction process completed.")

def run_oh1_import():
    """
    The program will run these programs in order:

        extract_and_ingest_text_from_oh1_pdf.py

    """
    print("Extracting text from OH1 files...")

    run_bin_program("extract_and_ingest_text_from_oh1_pdf.py")

    print("Data extraction and prediction process completed.")

def run_assemble_text():
    """
    The program will run these programs in order:

        assemble_text.py

    """
    print("Assembling extracted text into a single file...")

    run_bin_program("assemble_text.py")

    print("Text assembly process completed.")

if __name__ == "__main__":
    run_all()
