"use strict";

// These functions are for the drag-and-drop UI
function drag() {
    document.getElementById('uploadFile').parentNode.className = 'draging dragBox';
}
function drop() {
    document.getElementById('uploadFile').parentNode.className = 'dragBox';
}

// This is the main function that handles the upload AND the API call
function handleFileUpload(event) {
    var file = event.target.files[0];
    if (!file) {
        return;
    }

    // --- NEW VALIDATION CODE ---
    var MAX_FILE_SIZE_MB = 10;
    var MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
    var ALLOWED_TYPES = ['image/jpeg', 'image/png'];

    // 1. File Type Check (in case drag-and-drop bypasses HTML accept attribute)
    if (ALLOWED_TYPES.indexOf(file.type) === -1) {
        alert("Invalid file type. Please upload a JPEG or PNG image.");
        // Clear the file input for re-selection
        event.target.value = null; 
        return;
    }

    // 2. File Size Check
    if (file.size > MAX_FILE_SIZE_BYTES) {
        alert("File is too large. Maximum size is " + MAX_FILE_SIZE_MB + "MB.");
        event.target.value = null;
        return;
    }
    // --- END NEW VALIDATION CODE ---


    // --- NEW CODE TO SWAP UPLOAD AREA FOR NEW BUTTON ---
    var uploadArea = document.getElementById("uploadArea");
    var newDiagnosisBtnWrapper = document.getElementById("newDiagnosisWrapper");

    uploadArea.style.display = 'none'; // Hide the upload area
    newDiagnosisBtnWrapper.style.display = 'block'; // Show the "+ New Diagnosis" button
    // --- END NEW CODE ---


    // --- 1. Show Image Preview ---
    var fileName = URL.createObjectURL(file);
    var preview = document.getElementById("preview");
    var previewImg = document.createElement("img");
    previewImg.setAttribute("src", fileName);
    preview.innerHTML = ""; // Clear previous preview
    preview.appendChild(previewImg);
    
    // --- 2. Prepare for API Call ---
    var resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "<p class='loading-message'>Diagnosing... Please wait.</p>";
    
    // Get the new wrapper container
    var container = document.getElementById("diagnosisContainer");
    
    // Un-hide the *entire container*
    container.style.display = 'flex'; 

    var formData = new FormData();
    formData.append("file", file); // "file" must match the key in our Flask app

    // --- 3. Send Image to Backend using fetch() ---
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json()) // Get the JSON response from Flask
    .then(data => {
        // --- 4. Display the Results ---
        console.log(data); // Log the data to the console for debugging
        
        if (data.error) {
            // Display the specific error message sent from the Python backend (app.py)
            resultsDiv.innerHTML = `<p class='error-message'>Error: ${data.error}</p>`;
        } else {
            var confidence = (data.confidence * 100).toFixed(2);
            var resultHTML = `
                <h3>Diagnosis Result</h3>
                <p><strong>Predicted Class:</strong> <span class="diagnosis-class">${data.predicted_class}</span></p>
                <p><strong>Confidence:</strong> <span class="diagnosis-confidence">${confidence}%</span></p>
                <h4>All Probabilities:</h4>
                <ul>
            `;
            
            for (const [className, prob] of Object.entries(data.all_probabilities)) {
                resultHTML += `<li><span>${className}:</span> <strong>${(prob * 100).toFixed(2)}%</strong></li>`;
            }
            
            resultHTML += "</ul>";
            resultsDiv.innerHTML = resultHTML;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultsDiv.innerHTML = `<p class='error-message'>An error occurred. Could not connect to the server or retrieve a response.</p>`;
    });
}