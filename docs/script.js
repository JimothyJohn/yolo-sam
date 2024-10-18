// AI-generated comment: Main script for handling image upload and object detection

// Constants for API endpoint and image size used in detection
const API_ENDPOINT = "https://yolo.advin.io/Prod/detect";
const DETECTION_IMAGE_SIZE = 640; // AI-generated comment: Image size used in the detection model

// Elements from the DOM
const imageUpload = document.getElementById('imageUpload');
const submitBtn = document.getElementById('submitBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// AI-generated comment: Global variable to hold the uploaded image
let uploadedImage = null;

// AI-generated comment: Event listener for image upload
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        // AI-generated comment: Create an Image object to load the uploaded file
        const img = new Image();
        img.onload = () => {
            // AI-generated comment: Set canvas dimensions to match the image
            canvas.width = img.width;
            canvas.height = img.height;
            // AI-generated comment: Draw the image onto the canvas
            ctx.drawImage(img, 0, 0);
            uploadedImage = img;
        };
        // AI-generated comment: Read the image file as a Data URL
        const reader = new FileReader();
        reader.onload = (e) => {
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// AI-generated comment: Event listener for the submit button
submitBtn.addEventListener('click', () => {
    if (!uploadedImage) {
        alert('Please upload an image first.');
        return;
    }

    // AI-generated comment: Create a hidden canvas to resize the image for detection
    const hiddenCanvas = document.createElement('canvas');
    hiddenCanvas.width = DETECTION_IMAGE_SIZE;
    hiddenCanvas.height = DETECTION_IMAGE_SIZE;
    const hiddenCtx = hiddenCanvas.getContext('2d');
    // AI-generated comment: Draw the resized image onto the hidden canvas
    hiddenCtx.drawImage(uploadedImage, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    // AI-generated comment: Get the base64-encoded image data
    const imageBase64 = hiddenCanvas.toDataURL('image/jpeg').split(',')[1];

    // AI-generated comment: Prepare the payload for the API request
    const payload = {
        image: imageBase64,
        conf_thres: 0.5,
        iou_thres: 0.5
    };

    // AI-generated comment: Send a POST request to the API endpoint
    fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        // AI-generated comment: Check for detections in the response
        if (data.detections) {
            drawDetections(data.detections);
        } else {
            alert('No detections found.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during detection.');
    });
});

// AI-generated comment: Function to draw bounding boxes on the canvas
function drawDetections(detections) {
    // AI-generated comment: Redraw the original image
    ctx.drawImage(uploadedImage, 0, 0);

    // AI-generated comment: Calculate scaling factors between detection size and original image size
    const scaleX = canvas.width / DETECTION_IMAGE_SIZE;
    const scaleY = canvas.height / DETECTION_IMAGE_SIZE;

    // AI-generated comment: Set styles for bounding boxes and labels
    ctx.lineWidth = 4;
    ctx.font = '20px Arial';
    ctx.textBaseline = 'top';

    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;

        // AI-generated comment: Scale bounding box coordinates to the canvas size
        const rectX = x1 * scaleX;
        const rectY = y1 * scaleY;
        const rectWidth = (x2 - x1) * scaleX;
        const rectHeight = (y2 - y1) * scaleY;

        // AI-generated comment: Draw bounding box
        ctx.strokeStyle = 'green';
        ctx.strokeRect(rectX, rectY, rectWidth, rectHeight);

        // AI-generated comment: Prepare label text
        const label = `${detection.class_name} ${detection.score.toFixed(2)}`;
        const textWidth = ctx.measureText(label).width;
        const textHeight = parseInt(ctx.font, 10);

        // AI-generated comment: Draw label background
        ctx.fillStyle = 'green';
        ctx.fillRect(rectX, rectY, textWidth + 6, textHeight + 4);

        // AI-generated comment: Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(label, rectX + 3, rectY + 2);
    });
}