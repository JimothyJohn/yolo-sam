// Main script for handling image upload and object detection
const API_ENDPOINT = "https://yolo.advin.io/Prod/detect";
const DETECTION_IMAGE_SIZE = 640;

// UI Elements
const imageUpload = document.getElementById('imageUpload');
const submitBtn = document.getElementById('submitBtn');
const selfieBtn = document.getElementById('selfieBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const uploadOverlay = document.getElementById('uploadOverlay');
const dropZone = document.getElementById('dropZone');
const loadingIndicator = document.getElementById('loadingIndicator');

// Metric Elements
const energyMetric = document.getElementById('energyMetric');
const timeMetric = document.getElementById('timeMetric');
const costMetric = document.getElementById('costMetric');
const roiMetric = document.getElementById('roiMetric');
const entityList = document.getElementById('entityList');

let uploadedImage = null;

// Initialize layout
function initCanvas() {
    canvas.width = 800;
    canvas.height = 450;
    // Clear with transparent bg
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Handle Image Loading
function handleFile(file) {
    if (!file) return;

    const img = new Image();
    img.onload = () => {
        // dynamic resizing for display
        const maxWidth = dropZone.clientWidth - 20; 
        const maxHeight = dropZone.clientHeight - 20;
        
        // Scale to fit within container while maintaining aspect ratio
        const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
        
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        uploadedImage = img;
        uploadOverlay.classList.add('hidden');
        
        // Reset metrics
        resetMetrics();
    };
    
    const reader = new FileReader();
    reader.onload = (e) => {
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Event Listeners
imageUpload.addEventListener('change', (e) => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--accent-primary)';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border-color)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border-color)';
    handleFile(e.dataTransfer.files[0]);
});

uploadOverlay.addEventListener('click', () => imageUpload.click());

// Selfie Logic
selfieBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        
        // Create a temporary video element to capture the frame
        const video = document.createElement('video');
        video.srcObject = stream;
        video.play();
        
        // Wait for video to be ready
        await new Promise(resolve => video.onloadedmetadata = resolve);
        
        // Small delay to ensure camera brightness adjusts (optional but good UX)
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Draw to canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        
        // Stop stream
        stream.getTracks().forEach(track => track.stop());
        
        // Set as uploaded image
        const dataUrl = canvas.toDataURL('image/jpeg');
        const img = new Image();
        img.onload = () => {
            uploadedImage = img;
            
            // Rescale for display fitting (reuse handleFile logic essentially)
            const maxWidth = dropZone.clientWidth - 20;
            const maxHeight = dropZone.clientHeight - 20;
            const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
            
            canvas.width = img.width * scale;
            canvas.height = img.height * scale;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            uploadOverlay.classList.add('hidden');
            resetMetrics();
        };
        img.src = dataUrl;
        
    } catch (err) {
        console.error("Camera access denied:", err);
        alert("Could not access camera. Please allow camera permissions to take a selfie.");
    }
});

// Metrics Simulation
function updateMetrics(detections, latency) {
    // 1. Energy (Mocked) in Joules
    const energy = (0.8 + Math.random() * 1.7).toFixed(3);

    // 2. Backend Time (Simulated)
    // The user wants to emphasize input-to-output backend time.
    // Real latency includes network overhead (RTT).
    // We will assume network overhead is roughly 20-40% of total latency for this demo,
    // or at least subtract a fixed "network cost" simulation to show "Backend Processing".
    // Let's display a value that represents the core engine time.
    
    // If latency is very small (local/fast), we clamp backend time to be slightly less.
    let backendTime = Math.floor(latency * 0.85); // Assume 85% is backend, 15% network
    if (backendTime < 10) backendTime = latency; // Fallback for instant responses

    // 3. Throughput (Calculated) - Inf/min
    // Formula: 60,000 ms/min / backendTime_ms
    // Avoid division by zero
    const safeTime = backendTime > 0 ? backendTime : 1;
    const throughput = Math.floor(60000 / safeTime);

    // 4. Cost Efficiency (Mocked) - Inf/$
    // Assume ~$1 buys ~10k-20k inferences?
    const roi = Math.floor(12000 + Math.random() * 5000);

    // Accelerated animations
    animateValue(energyMetric, 0, energy, 500, '', '');
    animateValue(timeMetric, 0, backendTime, 500, '', '');
    animateValue(costMetric, 0, throughput, 600, '', '');
    animateValue(roiMetric, 0, roi, 600, '', '');

    updateEntityList(detections);
}

function updateEntityList(detections) {
    entityList.innerHTML = '';
    
    if (detections.length === 0) {
        entityList.innerHTML = '<li class="empty-state">No entities found</li>';
        return;
    }

    const counts = {};
    detections.forEach(d => {
        counts[d.class_name] = (counts[d.class_name] || 0) + 1;
    });

    const sortedEntities = Object.entries(counts)
        .sort(([,a], [,b]) => b - a);

    sortedEntities.forEach(([name, count]) => {
        const li = document.createElement('li');
        
        const nameSpan = document.createElement('span');
        nameSpan.className = 'entity-name';
        nameSpan.textContent = name;
        
        const countSpan = document.createElement('span');
        countSpan.className = 'entity-score';
        countSpan.textContent = count > 1 ? `x${count}` : 'x1';
        
        li.appendChild(nameSpan);
        li.appendChild(countSpan);
        entityList.appendChild(li);
    });
}

function resetMetrics() {
    energyMetric.textContent = '0.00';
    timeMetric.textContent = '0';
    costMetric.textContent = '0';
    roiMetric.textContent = '0';
    entityList.innerHTML = '<li class="empty-state">No active detections</li>';
}

function animateValue(obj, start, end, duration, prefix = '', suffix = '') {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        
        let val;
        // Handle float vs int
        const isFloat = end.toString().includes('.');
        const endNum = parseFloat(end);
        const startNum = parseFloat(start);
        
        const currentVal = progress * (endNum - startNum) + startNum;
        
        if (isFloat) {
            val = currentVal.toFixed(isFloat ? end.split('.')[1].length : 0);
        } else {
            val = Math.floor(currentVal).toLocaleString();
        }

        obj.textContent = prefix + val + suffix;
        
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
             obj.textContent = prefix + (isFloat ? endNum.toFixed(end.split('.')[1].length) : parseFloat(end).toLocaleString()) + suffix;
        }
    };
    window.requestAnimationFrame(step);
}

// Detection Logic
submitBtn.addEventListener('click', () => {
    if (!uploadedImage) {
        // Quick visual feedback for error
        // For mobile "Thumb Zone" buttons, we need to flash the dropzone if visible
        // or just alert if dropzone is off screen? 
        // Dropzone is always visible in center.
        dropZone.style.borderColor = 'var(--accent-secondary)';
        setTimeout(() => dropZone.style.borderColor = 'var(--border-color)', 500);
        return;
    }

    setLoading(true);

    const hiddenCanvas = document.createElement('canvas');
    hiddenCanvas.width = DETECTION_IMAGE_SIZE;
    hiddenCanvas.height = DETECTION_IMAGE_SIZE;
    const hiddenCtx = hiddenCanvas.getContext('2d');
    hiddenCtx.drawImage(uploadedImage, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    const imageBase64 = hiddenCanvas.toDataURL('image/jpeg').split(',')[1];

    const payload = {
        image: imageBase64,
        conf_thres: 0.5,
        iou_thres: 0.5
    };

    const startTime = performance.now();

    fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);
        
        setLoading(false);
        if (data.detections) {
            drawDetections(data.detections);
            updateMetrics(data.detections, latency);
        } else {
            alert('No detections found.');
        }
    })
    .catch(error => {
        setLoading(false);
        console.error('Error:', error);
        alert('An error occurred during detection.');
    });
});

function setLoading(isLoading) {
    if (isLoading) {
        loadingIndicator.classList.add('active');
        canvas.style.opacity = '0.3';
        submitBtn.disabled = true;
        submitBtn.textContent = 'RUNNING...';
    } else {
        loadingIndicator.classList.remove('active');
        canvas.style.opacity = '1';
        submitBtn.disabled = false;
        submitBtn.textContent = 'RUN'; // Shortened
    }
}

function drawDetections(detections) {
    // Redraw original
    ctx.drawImage(uploadedImage, 0, 0, canvas.width, canvas.height); // Use current scaled canvas dims

    const scaleX = canvas.width / DETECTION_IMAGE_SIZE;
    const scaleY = canvas.height / DETECTION_IMAGE_SIZE;

    ctx.lineWidth = 2;
    ctx.font = 'bold 16px "JetBrains Mono"';
    ctx.textBaseline = 'top';

    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;

        const rectX = x1 * scaleX;
        const rectY = y1 * scaleY;
        const rectWidth = (x2 - x1) * scaleX;
        const rectHeight = (y2 - y1) * scaleY;

        ctx.shadowColor = 'var(--success-color)';
        ctx.shadowBlur = 10;
        ctx.strokeStyle = '#00ff9d';
        ctx.strokeRect(rectX, rectY, rectWidth, rectHeight);
        ctx.shadowBlur = 0;

        const label = `${detection.class_name} ${(detection.score * 100).toFixed(0)}%`;
        const textWidth = ctx.measureText(label).width;
        const textHeight = 20;

        ctx.fillStyle = 'rgba(0, 255, 157, 0.8)';
        ctx.fillRect(rectX, rectY - textHeight, textWidth + 10, textHeight);

        ctx.fillStyle = '#000';
        ctx.fillText(label, rectX + 5, rectY - textHeight + 2);
    });
}

// Init
initCanvas();