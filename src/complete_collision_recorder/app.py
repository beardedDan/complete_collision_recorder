# complete_collision_recorder/webapp/app.py

import os
import sys
from flask import Flask, render_template, request, jsonify
import complete_collision as cc

app = Flask(__name__)

# Route to render the main HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Renders the HTML file

# API route to handle JavaScript requests
@app.route('/record_collision', methods=['POST'])
def record_collision():
    data = request.json  # Get data sent by JavaScript
    # Call your collision recording function here, e.g., `your_python_code.record_collision(data)`
    response_data = {"status": "Collision recorded", "data": data}
    return jsonify(response_data)  # Send back a response

if __name__ == '__main__':
    app.run(port=5000, debug=True)