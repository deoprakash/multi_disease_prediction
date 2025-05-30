// Prevents default drag events
document.getElementById('upload-area').addEventListener('click', function() {
    document.getElementById('file-input').click();
  });
  
  document.getElementById('file-input').addEventListener('change', function() {
    if(this.files && this.files[0]) {
      document.getElementById('file-name').textContent = this.files[0].name;
    }
  });
  
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    document.getElementById('upload-area').addEventListener(eventName, preventDefaults, false);
  });
  
  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }
  
  document.getElementById('upload-area').addEventListener('drop', handleDrop, false);
  
  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    document.getElementById('file-input').files = files;
    if(files && files[0]) {
      document.getElementById('file-name').textContent = files[0].name;
    }
  }
  
  // Camera feature
  const cameraInterface = document.getElementById('camera-interface');
  const video = document.getElementById('video');
  const captureBtn = document.getElementById('capture-btn');
  const useCameraBtn = document.getElementById('use-camera-btn');
  const uploadArea = document.getElementById('upload-area');
  let stream;

  useCameraBtn.addEventListener('click', function() {
    cameraInterface.style.display = 'block';
    uploadArea.style.display = 'none';
    
    // Start camera
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function(s) {
        stream = s;
        video.srcObject = stream;
      })
      .catch(function(err) {
        console.log("Error accessing camera: " + err);
        alert("Unable to access camera. Please ensure you've granted camera permissions.");
        cameraInterface.style.display = 'none';
        uploadArea.style.display = 'block';
      });
  });

  captureBtn.addEventListener('click', function() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert the captured image to a file and update the input field
    canvas.toBlob(function(blob) {
      const file = new File([blob], "captured_image.jpg", { type: "image/jpeg" });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      document.getElementById('file-input').files = dataTransfer.files;
      document.getElementById('file-name').textContent = file.name;
      
      // Stop the camera stream
      stream.getTracks().forEach(track => track.stop());
      cameraInterface.style.display = 'none';
      uploadArea.style.display = 'block';
    });
  });

  // Handle file selection and display filename
  document.getElementById('file-input').addEventListener('change', function(event) {
    const fileName = event.target.files[0] ? event.target.files[0].name : '';
    document.getElementById('file-name').innerText = fileName;
  });

  // Toggle camera functionality
  document.getElementById('use-camera-btn').addEventListener('click', async function() {
    const videoElement = document.getElementById('camera-preview');
    const cameraContainer = document.getElementById('camera-container');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });

    // Show camera preview
    videoElement.srcObject = stream;
    cameraContainer.style.display = 'block';
    document.getElementById('use-camera-btn').style.display = 'none';

    // Capture image on button press (can be customized to trigger prediction)
    videoElement.addEventListener('click', function() {
      alert('Capture the image and analyze');
    });
  });

  // Close the camera view
  document.getElementById('close-camera-btn').addEventListener('click', function() {
    const cameraContainer = document.getElementById('camera-container');
    const videoElement = document.getElementById('camera-preview');
    videoElement.srcObject.getTracks().forEach(track => track.stop()); // Stop video stream
    cameraContainer.style.display = 'none';
    document.getElementById('use-camera-btn').style.display = 'block';
  });

  window.addEventListener("load", function () {
  const navType = performance.getEntriesByType("navigation")[0].type;

  if (navType === "reload" || navType === "navigate" || navType === "back_forward") {
    localStorage.removeItem("predictionResult");
    sessionStorage.removeItem("predictionResult");
  }
});

function clearPredictionAndImage() {
  const predictionOutput = document.getElementById("prediction-output");
  if (predictionOutput) {
    predictionOutput.innerText = "";
  }

  const imageDisplay = document.getElementById("image-preview");
  if (imageDisplay) {
    imageDisplay.src = "";
    imageDisplay.style.display = "none";
  }

  const fileInput = document.getElementById("imageUpload");
  if (fileInput) {
    fileInput.value = "";
  }
}

// Clear on full load (refresh or first visit)
window.addEventListener("DOMContentLoaded", clearPredictionAndImage);

// Clear on back/forward cache navigation
window.addEventListener("pageshow", function (event) {
  if (event.persisted) {
    clearPredictionAndImage();
  }
});

