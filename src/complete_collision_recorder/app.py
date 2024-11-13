# complete_collision_recorder/webapp/app.py

import os
from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

# Route to render the main HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Renders the HTML file


@app.route('/extract-cad-text', methods=['GET'])
def extract_cad_text():
    print("Extracting text from CAD files...")
    try:
        script_path = os.path.join(os.getcwd(), 'extract_and_ingest_text_from_cad_pdf.py')
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'status': 'error', 'message': result.stderr}), 500
        return jsonify({'status': 'success', 'message': result.stdout}), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    print("Data extracted from CAD files.")

@app.route('/extract-oh1-text', methods=['GET'])
def extract_oh1_text():
    print("Extracting text from OH1 files...")
    try:
        script_path = os.path.join(os.getcwd(), 'extract_and_ingest_text_from_oh1_pdf.py')
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'status': 'error', 'message': result.stderr}), 500
        return jsonify({'status': 'success', 'message': result.stdout}), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    print("Data extracted from OH1 PDF files.")

@app.route('/assemble-text', methods=['GET'])
def assemble_text():
    print("Assembling extracted text into a single file...")

    try:
        script_path = os.path.join(os.getcwd(), 'assemble_text.py')
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'status': 'error', 'message': result.stderr}), 500
        return jsonify({'status': 'success', 'message': result.stdout}), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    print("Text assembly process completed.")

@app.route('/score-and-describe-text', methods=['GET'])
def score_and_describe_text():
    print("Scoring and describing text...")

    try:
        script_path = os.path.join(os.getcwd(), 'score_and_describe_reports.py')
        # result = subprocess.run(['python', script_path], capture_output=True, text=True)
        result = subprocess.run(['python', 'extract_and_ingest_text_from_cad_pdf.py', '--testing'], capture_output=True, text=True)

        if result.returncode != 0:
            return jsonify({'status': 'error', 'message': result.stderr}), 500
        return jsonify({'status': 'success', 'message': result.stdout}), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    print("Reports scored and described.")    

@app.route('/run-all', methods=['GET'])
def run_all():
    try:
        script_path = os.path.join(os.getcwd(), 'extract_and_ingest_text_from_cad_pdf.py')
        script_path = os.path.join(os.getcwd(), 'extract_and_ingest_text_from_oh1_pdf.py')
        script_path = os.path.join(os.getcwd(), 'assemble_text.py')
        script_path = os.path.join(os.getcwd(), 'score_and_describe_reports.py')
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'status': 'error', 'message': result.stderr}), 500
        return jsonify({'status': 'success', 'message': result.stdout}), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    print("Reports scored and described.")    


if __name__ == '__main__':
    app.run(port=5000, debug=True)