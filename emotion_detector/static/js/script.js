$(document).ready(function() {
    // Drag and drop functionality
    const uploadArea = $('#uploadArea');
    const imageInput = $('#imageInput');
    const imagePreview = $('#imagePreview');
    const resultSection = $('#resultSection');
    const emptyState = $('#emptyState');
    
    // File upload handling
    imageInput.on('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleImageUpload(file);
        }
    });
    
    // Drag and drop
    uploadArea.on('dragover', function(e) {
        e.preventDefault();
        uploadArea.css('background', '#e9ecef');
    });
    
    uploadArea.on('dragleave', function() {
        uploadArea.css('background', '#f8f9fa');
    });
    
    uploadArea.on('drop', function(e) {
        e.preventDefault();
        uploadArea.css('background', '#f8f9fa');
        const file = e.originalEvent.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageUpload(file);
        } else {
            alert('Please drop an image file');
        }
    });
    
    // Click to upload
    uploadArea.on('click', function() {
        imageInput.click();
    });
    
    function handleImageUpload(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            // Show preview
            imagePreview.attr('src', e.target.result);
            imagePreview.show();
            
            // Hide empty state, show results
            emptyState.hide();
            resultSection.show();
            
            // Show loading
            $('#emotionResults').html(`
                <div class="text-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p>Analyzing emotions...</p>
                </div>
            `);
            
            // Send to server for prediction
            predictImage(file);
        }
        
        reader.readAsDataURL(file);
    }
    
    function predictImage(file) {
        const formData = new FormData();
        formData.append('image', file);
        
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                if (response.success) {
                    displayResults(response);
                } else {
                    alert('Error: ' + response.error);
                }
            },
            error: function() {
                alert('Error connecting to server');
            }
        });
    }
    
    function displayResults(data) {
        // Show dominant emotion
        $('#dominantEmotion').html(`
            <div class="dominant-emotion">
                ${data.dominant_emotion}<br>
                <small>Confidence: ${data.confidence.toFixed(2)}%</small>
            </div>
        `);
        
        // Show all emotions as progress bars
        let html = '';
        data.predictions.forEach(emotion => {
            html += `
                <div class="emotion-bar">
                    <div class="d-flex justify-content-between">
                        <span class="emotion-label">${emotion.emotion}</span>
                        <span>${emotion.probability.toFixed(2)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar" 
                             role="progressbar" 
                             style="width: ${emotion.probability}%; background-color: ${emotion.color}"
                             aria-valuenow="${emotion.probability}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                </div>
            `;
        });
        
        $('#emotionResults').html(html);
    }
});