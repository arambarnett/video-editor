console.log('Script loaded!'); // Add this at the top

$(document).ready(function () {
  console.log('Document ready!'); // Add this to verify jQuery is working

  let selectedStyle = '';
  let downloadUrl = '';

  $("#uploadBtn").click(function () {
    console.log('Button clicked, current state:', {  // Debug log
        isState4: $(this).find('.state-4-text').hasClass('active'),
        downloadUrl: downloadUrl
    });
    
    if ($(this).find('.state-4-text').hasClass('active') && downloadUrl) {
        console.log('Initiating download with URL:', downloadUrl); // Debug log
        window.location.href = downloadUrl;
    } else if ($(this).find('.state-1-text').hasClass('active')) {
        $("#fileInput").click();
    }
  });

  $("#fileInput").change(function (e) {
    const files = e.target.files;
    if (files.length > 0) {
      // Show selected files count
      let fileNames = Array.from(files).map(file => file.name).join(', ');
      $('#fileList').html(`Selected files: ${fileNames}`);
      
      $(".primary-btn").removeClass("state-1").addClass("state-2");
      $(".state-1-text").removeClass("active");
      $(".state-2-text").addClass("active");
      $(".file-details").show();
      $("#styleOptions").show();
    }
  });

  $(".feature").click(function () {
    selectedStyle = $(this).data('style');
    $(".feature").removeClass("selected");
    $(this).addClass("selected");
    $(".selected-style").text(`Selected style: ${selectedStyle}`);
    $("#styleOptions").hide();
    uploadFiles();
  });

  async function uploadFiles() {
    const fileInput = document.querySelector('input[type="file"]');
    const styleSelect = document.querySelector('select[name="style"]');
    const processingStatus = document.getElementById('processingStatus');
    const downloadContainer = document.getElementById('downloadContainer');
    
    if (!fileInput.files.length) {
        alert('Please select files to upload');
        return;
    }
    
    const formData = new FormData();
    for (let file of fileInput.files) {
        formData.append('files', file);
    }
    formData.append('style', styleSelect.value);
    
    try {
        processingStatus.textContent = 'Processing...';
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        processingStatus.textContent = data.message;
        
        // Construct download URL using session info
        const downloadUrl = `/download/${data.session_id}/${data.timestamp}`;
        
        // Create download link
        const downloadLink = document.createElement('a');
        downloadLink.href = downloadUrl;
        downloadLink.textContent = 'Download Edited Video';
        downloadLink.className = 'download-button';
        downloadContainer.innerHTML = ''; // Clear any existing content
        downloadContainer.appendChild(downloadLink);
        
    } catch (error) {
        console.error('Upload error:', error);
        processingStatus.textContent = 'Error: ' + error.message;
    }
  }

  $('#startOverBtn').click(function() {
    // Reset UI
    $('.primary-btn').removeClass('state-2 state-3 state-4').addClass('state-1');
    $('.state-1-text').addClass('active');
    $('.state-2-text, .state-3-text, .state-4-text').removeClass('active');
    $('#styleOptions').hide();
    $('#startOverBtn').hide();
    $('.file-details').hide();
    $('#progressBar').hide();
    $('#progressText').hide();
    $('#fileInput').val('');
    selectedStyle = '';
    downloadUrl = ''; // Clear the download URL
  });

  async function uploadVideos(files, style) {
    try {
        const formData = new FormData();
        
        // Add files
        for (const file of files) {
            formData.append('files', file);
        }
        
        // Add style
        formData.append('style', style);

        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        console.log('Server response:', data); // Debug log

        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        // Show results
        const resultDiv = document.querySelector('.file-details');
        resultDiv.innerHTML = ''; // Clear previous results
        resultDiv.classList.add('active');

        // Add download button
        const downloadBtn = document.createElement('a');
        downloadBtn.href = `/api/download/${data.session_id}`;
        downloadBtn.className = 'primary-btn';
        downloadBtn.innerHTML = `
            <span class="effect-1"></span>
            <span class="effect-2"></span>
            <span class="text">Download Edited Video</span>
        `;
        resultDiv.appendChild(downloadBtn);

        // Add analysis if available
        if (data.analysis) {
            const analysisDiv = document.createElement('div');
            analysisDiv.className = 'analysis-container';
            analysisDiv.innerHTML = `
                <h3>Video Analysis</h3>
                <div class="transcription">
                    <p>${data.analysis.full_text}</p>
                </div>
                <div class="key-points">
                    <h4>Key Points</h4>
                    <ul>
                        ${data.analysis.key_points.map(point => `<li>${point}</li>`).join('')}
                    </ul>
                </div>
            `;
            resultDiv.appendChild(analysisDiv);
        }

        // Update UI state
        const uploadBtn = document.querySelector('.primary-btn-wrapper');
        uploadBtn.classList.remove('uploading');
        uploadBtn.classList.add('active');

        return data;

    } catch (error) {
        console.error('Error:', error);
        // Show error state
        const uploadBtn = document.querySelector('.primary-btn-wrapper');
        uploadBtn.classList.remove('uploading');
        uploadBtn.classList.add('error');
        throw error;
    }
  }

  // Form submit handler
  document.querySelector('form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const uploadBtn = document.querySelector('.primary-btn-wrapper');
    uploadBtn.classList.add('uploading');
    uploadBtn.classList.remove('error', 'active');
    
    try {
        const fileInput = document.querySelector('input[type="file"]');
        const styleSelect = document.querySelector('select[name="style"]');
        
        await uploadVideos(fileInput.files, styleSelect.value);
        
    } catch (error) {
        console.error('Upload failed:', error);
        alert(`Error: ${error.message}`);
    }
  });

  function displayResults(data) {
    if (!data.success) {
        showError(data.error || 'An error occurred during processing');
        return;
    }

    const resultDiv = document.getElementById('results');
    if (!resultDiv) return;

    // Clear previous results
    resultDiv.innerHTML = '';

    // Create analysis section
    const analysisDiv = document.createElement('div');
    analysisDiv.className = 'analysis-section';

    // Add transcription
    if (data.analysis && data.analysis.full_text) {
        const transcriptDiv = document.createElement('div');
        transcriptDiv.className = 'transcript';
        transcriptDiv.innerHTML = `
            <h3>Transcription</h3>
            <p>${data.analysis.full_text}</p>
        `;
        analysisDiv.appendChild(transcriptDiv);
    }

    // Add key points
    if (data.analysis && data.analysis.key_points) {
        const pointsDiv = document.createElement('div');
        pointsDiv.className = 'key-points';
        pointsDiv.innerHTML = `
            <h3>Key Points</h3>
            <ul>
                ${data.analysis.key_points.map(point => `<li>${point}</li>`).join('')}
            </ul>
        `;
        analysisDiv.appendChild(pointsDiv);
    }

    // Add download button
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'download-button';
    downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Edited Video';
    downloadBtn.onclick = () => downloadResults(data.session_id);
    analysisDiv.appendChild(downloadBtn);

    // Add to results
    resultDiv.appendChild(analysisDiv);
    resultDiv.style.display = 'block';
  }

  function showError(message) {
    const resultDiv = document.getElementById('results');
    if (!resultDiv) return;

    resultDiv.innerHTML = `
        <div class="error-message">
            <i class="fas fa-exclamation-circle"></i>
            <p>${message}</p>
        </div>
    `;
    resultDiv.style.display = 'block';
  }

  function downloadResults(sessionId) {
    const downloadUrl = `/api/download/${sessionId}`;
    
    // Show loading state
    const downloadBtn = document.querySelector('.download-button');
    downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Preparing Download...';
    
    // Trigger download
    fetch(downloadUrl)
        .then(response => {
            if (!response.ok) throw new Error('Download failed');
            return response.blob();
        })
        .then(blob => {
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `edited_video_package_${sessionId}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            
            // Reset button state
            downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Edited Video';
        })
        .catch(error => {
            console.error('Download error:', error);
            downloadBtn.innerHTML = '<i class="fas fa-exclamation-circle"></i> Download Failed';
        });
  }
});

// Example frontend code
function getVideoUrl(filename) {
    return `/output/${filename}`;
}
