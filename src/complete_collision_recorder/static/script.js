// complete_collision_recorder/webapp/static/script.js

function sendDataToPython(data) {
    fetch('/record_collision', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

// Example: Trigger this function on button click
document.getElementById("recordButton").onclick = function() {
    const sampleData = { message: "Collision event data" };
    sendDataToPython(sampleData);
};