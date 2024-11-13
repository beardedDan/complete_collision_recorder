// complete_collision_recorder/webapp/static/script.js

// Define the functions for each button
function extractCadText() {
    alert('Extracting CAD Text.');
    // Make a GET request to the Flask route
    fetch('/extract-cad-text')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                alert('CAD text extraction successful: ' + data.message);
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            alert('Request failed: ' + error);
        });    
}

function extractOH1Text() {
    alert('Extracting OH1 Text.');
    fetch('/extract-oh1-text')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                alert('OH1 text extraction successful: ' + data.message);
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            alert('Request failed: ' + error);
        });        
}

function assembleText() {
    alert('Assembling Extracted Text.');
    fetch('/assemble-text')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                alert('Extracted text successfully assembled: ' + data.message);
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            alert('Request failed: ' + error);
        });     
}

function scoreAndDescribeText() {
    alert('Scoring and describing text.');
    fetch('/score-and-describe-text')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                alert('Extracted text successfully assembled: ' + data.message);
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            alert('Request failed: ' + error);
        });    
}

function runAll() {
    alert('Running entire import, assembly, and scoring process.');
    fetch('/run-all')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                alert('Process completed successfully: ' + data.message);
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            alert('Request failed: ' + error);
        });        
}

// Event listeners for button clicks
document.getElementById('extractCadTextButton').addEventListener('click', extractCadText);
document.getElementById('extractOH1TextButton').addEventListener('click', extractOH1Text);
document.getElementById('assembleTextButton').addEventListener('click', assembleText);
document.getElementById('scoreAndDescribeTextButton').addEventListener('click', scoreAndDescribeText);
document.getElementById('runAllButton').addEventListener('click', runAll);