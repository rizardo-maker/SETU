import os
import shutil
import base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from setu_pipeline import SETUPipeline
from config import BASE_DIR, OUTPUT_AUDIO_DIR, TEMP_DIR

app = FastAPI(title="SETU: Local Offline OCR-to-Speech Web UI")

# Mount output audio directory to serve generated .wav audio files
app.mount("/audio", StaticFiles(directory=str(OUTPUT_AUDIO_DIR)), name="audio")

# Initialize global pipeline instance
pipeline = SETUPipeline(auto_download_models=True)

class Base64FrameRequest(BaseModel):
    image_base64: str

@app.get("/", response_class=HTMLResponse)
def get_index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SETU | Real-Time Camera OCR to Speech</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-color: #0f172a;
                --card-bg: #1e293b;
                --accent-color: #38bdf8;
                --accent-hover: #0284c7;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --border-color: #334155;
                --success-bg: rgba(16, 185, 129, 0.1);
                --success-border: #10b981;
                --danger-bg: rgba(239, 68, 68, 0.1);
                --danger-border: #ef4444;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-main);
                min-height: 100vh;
                padding: 2rem;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            .header {
                text-align: center;
                margin-bottom: 2rem;
            }

            .header h1 {
                font-size: 2.2rem;
                font-weight: 700;
                background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }

            .header p {
                color: var(--text-muted);
                font-size: 1rem;
            }

            .container {
                width: 100%;
                max-width: 950px;
                display: grid;
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }

            .tabs {
                display: flex;
                gap: 1rem;
                margin-bottom: 1rem;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 0.5rem;
            }

            .tab-btn {
                background: none;
                border: none;
                color: var(--text-muted);
                font-size: 1.05rem;
                font-weight: 600;
                padding: 0.5rem 1rem;
                cursor: pointer;
                border-radius: 8px;
                transition: all 0.2s ease;
            }

            .tab-btn.active {
                color: var(--accent-color);
                background: rgba(56, 189, 248, 0.1);
            }

            .card {
                background-color: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            }

            .camera-container {
                position: relative;
                width: 100%;
                max-width: 640px;
                margin: 0 auto;
                border-radius: 12px;
                overflow: hidden;
                background: #000;
                border: 2px solid var(--border-color);
                min-height: 360px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            video {
                width: 100%;
                height: 360px;
                object-fit: cover;
                display: block;
            }

            .camera-hud {
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(15, 23, 42, 0.85);
                padding: 6px 14px;
                border-radius: 9999px;
                font-size: 0.85rem;
                font-weight: 600;
                color: var(--accent-color);
                border: 1px solid var(--border-color);
            }

            .cam-error-box {
                display: none;
                padding: 1rem;
                background-color: var(--danger-bg);
                border: 1px solid var(--danger-border);
                color: #fca5a5;
                border-radius: 8px;
                margin-top: 1rem;
                font-size: 0.9rem;
            }

            .select-input {
                background: #0f172a;
                color: var(--text-main);
                border: 1px solid var(--border-color);
                padding: 0.6rem 1rem;
                border-radius: 8px;
                font-size: 0.95rem;
                width: 100%;
                margin-bottom: 1rem;
            }

            .upload-zone {
                border: 2px dashed var(--accent-color);
                border-radius: 12px;
                padding: 2.5rem;
                text-align: center;
                cursor: pointer;
                transition: all 0.2s ease;
                background: rgba(56, 189, 248, 0.03);
            }

            .upload-zone:hover {
                background: rgba(56, 189, 248, 0.08);
                border-color: #60a5fa;
            }

            .upload-zone input {
                display: none;
            }

            .btn {
                background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
                color: #ffffff;
                border: none;
                padding: 0.8rem 1.5rem;
                font-size: 1rem;
                font-weight: 600;
                border-radius: 8px;
                cursor: pointer;
                transition: transform 0.1s ease, box-shadow 0.2s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
            }

            .btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
            }

            .btn-secondary {
                background: #334155;
                color: var(--text-main);
            }

            .btn-secondary:hover {
                background: #475569;
                box-shadow: 0 4px 12px rgba(71, 85, 105, 0.4);
            }

            .controls-row {
                display: flex;
                gap: 1rem;
                margin-top: 1rem;
            }

            .status-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.85rem;
                font-weight: 600;
                background-color: var(--success-bg);
                color: var(--success-border);
                border: 1px solid var(--success-border);
                margin-bottom: 1rem;
            }

            .text-box {
                background: #0f172a;
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 1rem;
                font-family: monospace;
                font-size: 0.95rem;
                white-space: pre-wrap;
                max-height: 250px;
                overflow-y: auto;
                color: #e2e8f0;
                margin-top: 0.5rem;
            }

            audio {
                width: 100%;
                margin-top: 0.75rem;
                border-radius: 8px;
            }

            .chunk-item {
                border-left: 3px solid var(--accent-color);
                background: #0f172a;
                padding: 1rem;
                border-radius: 0 8px 8px 0;
                margin-bottom: 1rem;
            }

            .loading {
                display: none;
                text-align: center;
                color: var(--accent-color);
                font-weight: 600;
                margin-top: 1rem;
            }

            .spinner {
                border: 4px solid rgba(255, 255, 255, 0.1);
                width: 36px;
                height: 36px;
                border-radius: 50%;
                border-left-color: var(--accent-color);
                animation: spin 1s linear infinite;
                margin: 0 auto 0.5rem auto;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📷 SETU Real-Time Camera OCR to Speech 🔊</h1>
            <p>Real-Time Camera Feed ──> RapidOCR / PaddleOCR ──> Text ──> Piper Neural Speech</p>
        </div>

        <div class="container">
            <div class="tabs">
                <button class="tab-btn active" id="tabCamBtn" onclick="switchTab('cam')">📷 Live Real-Time Camera</button>
                <button class="tab-btn" id="tabUploadBtn" onclick="switchTab('upload')">📁 Upload Document / Image</button>
            </div>

            <!-- Tab 1: Real-Time Camera Scanner -->
            <div class="card" id="camTabContent">
                <div style="margin-bottom: 0.75rem;">
                    <label style="font-size:0.85rem; color:var(--text-muted); display:block; margin-bottom:0.3rem;">Select Camera Device:</label>
                    <select id="cameraSelect" class="select-input" onchange="startCamera()"></select>
                </div>

                <div class="camera-container">
                    <video id="webcamVideo" autoplay playsinline muted></video>
                    <div class="camera-hud" id="cameraHud">🔴 LIVE CAMERA FEED</div>
                </div>
                <canvas id="captureCanvas" style="display:none;"></canvas>

                <div class="cam-error-box" id="camErrorBox"></div>

                <div class="controls-row">
                    <button class="btn" onclick="startCamera()" id="startCamBtn">▶️ Start Live Camera</button>
                    <button class="btn" onclick="captureAndProcessFrame()" id="captureBtn" style="flex:1;">📸 Capture & Read Text (OCR -> Speech)</button>
                </div>
                
                <div class="loading" id="camLoadingState">
                    <div class="spinner"></div>
                    <p>Processing camera frame: Running OCR -> Piper Speech...</p>
                </div>
            </div>

            <!-- Tab 2: File Upload Mode -->
            <div class="card" id="uploadTabContent" style="display:none;">
                <form id="uploadForm">
                    <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                        <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="margin: 0 auto 0.5rem auto; color: var(--accent-color);"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                        <p id="fileLabel" style="font-weight:600;">Click or Drag & Drop Image (.png, .jpg) or PDF (.pdf)</p>
                        <p style="font-size:0.85rem; color:var(--text-muted); margin-top:0.3rem;">Local offline processing</p>
                        <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.pdf,.bmp" onchange="updateFileLabel()">
                    </div>
                    <div class="controls-row">
                        <button type="submit" class="btn" style="flex:1;">⚡ Process Document & Synthesize Speech</button>
                    </div>
                </form>

                <div class="loading" id="loadingState">
                    <div class="spinner"></div>
                    <p>Running local OCR & generating Piper neural speech...</p>
                </div>
            </div>

            <!-- Assistive Voice Controls Card -->
            <div class="card" id="assistiveControlsCard">
                <h3>🎙️ Assistive Simulated Voice Controls</h3>
                <p style="color:var(--text-muted); font-size:0.9rem; margin-bottom:1rem;">Simulate blind user voice commands for chunked audio reading.</p>
                <div class="controls-row">
                    <button class="btn btn-secondary" onclick="triggerVoiceCmd('read_this')">🎙️ "Read This" (First Chunk)</button>
                    <button class="btn btn-secondary" onclick="triggerVoiceCmd('continue')">⏭️ "Continue" (Next Chunk)</button>
                </div>
                <div id="voiceCmdAudioContainer" style="margin-top:1rem;"></div>
            </div>

            <!-- Results Card -->
            <div class="card" id="resultCard" style="display:none;">
                <div class="status-badge" id="ocrBadge">OCR Completed</div>
                <h3 style="margin-bottom:0.5rem;">Pipeline Execution Summary</h3>
                <p id="metricsText" style="color:var(--text-muted); font-size:0.9rem; margin-bottom:1rem;"></p>

                <h4 style="margin-top:1rem;">Full Extracted Document Text</h4>
                <div class="text-box" id="fullTextDisplay"></div>

                <h4 style="margin-top:1.5rem; margin-bottom:0.5rem;">Synthesized Speech Audio</h4>
                <div id="chunksContainer"></div>
            </div>
        </div>

        <script>
            let cameraStream = null;

            async function enumerateCameras() {
                try {
                    const devices = await navigator.mediaDevices.enumerateDevices();
                    const videoDevices = devices.filter(d => d.kind === 'videoinput');
                    const select = document.getElementById('cameraSelect');
                    select.innerHTML = '';

                    if (videoDevices.length === 0) {
                        select.innerHTML = '<option value="">No camera devices found</option>';
                        return;
                    }

                    videoDevices.forEach((device, idx) => {
                        const option = document.createElement('option');
                        option.value = device.deviceId;
                        option.text = device.label || `Camera ${idx + 1}`;
                        select.appendChild(option);
                    });
                } catch (err) {
                    console.log("Enumerating devices error: ", err);
                }
            }

            function switchTab(tab) {
                if (tab === 'cam') {
                    document.getElementById('camTabContent').style.display = 'block';
                    document.getElementById('uploadTabContent').style.display = 'none';
                    document.getElementById('tabCamBtn').classList.add('active');
                    document.getElementById('tabUploadBtn').classList.remove('active');
                    startCamera();
                } else {
                    document.getElementById('camTabContent').style.display = 'none';
                    document.getElementById('uploadTabContent').style.display = 'block';
                    document.getElementById('tabUploadBtn').classList.add('active');
                    document.getElementById('tabCamBtn').classList.remove('active');
                    stopCamera();
                }
            }

            async function startCamera() {
                const errBox = document.getElementById('camErrorBox');
                errBox.style.display = 'none';
                errBox.innerHTML = '';

                stopCamera();

                const select = document.getElementById('cameraSelect');
                const selectedDeviceId = select.value;

                let constraints = { video: true };

                if (selectedDeviceId) {
                    constraints = { video: { deviceId: { exact: selectedDeviceId } } };
                }

                try {
                    const video = document.getElementById('webcamVideo');
                    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
                    video.srcObject = cameraStream;
                    await video.play();
                    
                    document.getElementById('startCamBtn').style.display = 'none';
                    document.getElementById('cameraHud').textContent = '🔴 LIVE CAMERA ACTIVE';

                    // Update camera labels if newly granted permission
                    await enumerateCameras();
                } catch (err) {
                    console.error("Camera access error:", err);
                    errBox.style.display = 'block';
                    errBox.innerHTML = `⚠️ <strong>Camera Error:</strong> ${err.name} - ${err.message}.<br>Please ensure camera access is allowed in your browser settings.`;
                    document.getElementById('startCamBtn').style.display = 'inline-flex';
                    document.getElementById('cameraHud').textContent = '⚠️ CAMERA OFFLINE';
                }
            }

            function stopCamera() {
                if (cameraStream) {
                    cameraStream.getTracks().forEach(track => track.stop());
                    cameraStream = null;
                }
                const video = document.getElementById('webcamVideo');
                if (video) {
                    video.srcObject = null;
                }
                document.getElementById('startCamBtn').style.display = 'inline-flex';
                document.getElementById('cameraHud').textContent = '⏸️ CAMERA PAUSED';
            }

            async function captureAndProcessFrame() {
                const video = document.getElementById('webcamVideo');
                if (!video.srcObject || video.paused || video.ended) {
                    await startCamera();
                }

                const canvas = document.getElementById('captureCanvas');
                const w = video.videoWidth || 1280;
                const h = video.videoHeight || 720;
                canvas.width = w;
                canvas.height = h;

                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, w, h);
                const dataUrl = canvas.toDataURL('image/png');

                document.getElementById('camLoadingState').style.display = 'block';
                document.getElementById('resultCard').style.display = 'none';

                try {
                    const response = await fetch('/api/process_base64_frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image_base64: dataUrl })
                    });

                    const data = await response.json();
                    document.getElementById('camLoadingState').style.display = 'none';

                    if (data.error) {
                        alert("Error processing camera frame: " + data.error);
                        return;
                    }

                    renderResults(data);
                } catch (err) {
                    document.getElementById('camLoadingState').style.display = 'none';
                    alert("Failed to process camera frame: " + err.message);
                }
            }

            function updateFileLabel() {
                const input = document.getElementById('fileInput');
                if (input.files.length > 0) {
                    document.getElementById('fileLabel').textContent = "Selected: " + input.files[0].name;
                }
            }

            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                if (fileInput.files.length === 0) {
                    alert("Please select an image or PDF document first.");
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                document.getElementById('loadingState').style.display = 'block';
                document.getElementById('resultCard').style.display = 'none';

                try {
                    const response = await fetch('/api/process', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    document.getElementById('loadingState').style.display = 'none';

                    if (data.error) {
                        alert("Error processing file: " + data.error);
                        return;
                    }

                    renderResults(data);
                } catch (err) {
                    document.getElementById('loadingState').style.display = 'none';
                    alert("Failed to process document: " + err.message);
                }
            });

            function renderResults(data) {
                document.getElementById('resultCard').style.display = 'block';
                document.getElementById('ocrBadge').textContent = `${data.file_type} | ${data.ocr_engine} | ${data.tts_engine}`;
                document.getElementById('metricsText').textContent = `Pages: ${data.num_pages} | OCR Time: ${data.metrics.ocr_time_seconds}s | TTS Time: ${data.metrics.tts_time_seconds}s | Total: ${data.metrics.total_time_seconds}s`;
                document.getElementById('fullTextDisplay').textContent = data.full_text || "[No text detected in camera frame]";

                const container = document.getElementById('chunksContainer');
                container.innerHTML = '';

                if (!data.chunks || data.chunks.length === 0) {
                    container.innerHTML = '<p style="color:var(--text-muted)">No text detected to synthesize speech.</p>';
                    return;
                }

                data.chunks.forEach((chunkText, idx) => {
                    const audioPath = data.audio_files[idx];
                    const filename = audioPath.split('/').pop();
                    
                    const div = document.createElement('div');
                    div.className = 'chunk-item';
                    div.innerHTML = `
                        <p style="font-weight:600; color:var(--accent-color); margin-bottom:0.3rem;">Chunk ${idx + 1} of ${data.chunks.length}</p>
                        <p style="font-size:0.95rem; margin-bottom:0.5rem;">${chunkText}</p>
                        <audio controls ${idx === 0 ? 'autoplay' : ''} src="/audio/${filename}"></audio>
                    `;
                    container.appendChild(div);
                });
            }

            async function triggerVoiceCmd(cmd) {
                try {
                    const response = await fetch('/api/voice_command/' + cmd, { method: 'POST' });
                    const res = await response.json();

                    if (res.error) {
                        alert(res.error);
                        return;
                    }

                    const filename = res.audio_file.split('/').pop();
                    const container = document.getElementById('voiceCmdAudioContainer');
                    container.innerHTML = `
                        <div class="chunk-item" style="border-left-color:#10b981;">
                            <p style="font-weight:600; color:#10b981; margin-bottom:0.3rem;">🎙️ Voice Cmd [Chunk ${res.chunk_idx}/${res.total_chunks}]</p>
                            <p style="font-size:0.95rem; margin-bottom:0.5rem;">"${res.text}"</p>
                            <audio controls autoplay src="/audio/${filename}"></audio>
                        </div>
                    `;
                } catch (err) {
                    alert("Error triggering voice command: " + err.message);
                }
            }

            window.addEventListener('DOMContentLoaded', async () => {
                await enumerateCameras();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/process")
async def process_file(file: UploadFile = File(...)):
    try:
        temp_path = TEMP_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = pipeline.process_file(temp_path, synthesize_audio=True)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/process_base64_frame")
async def process_base64_frame(req: Base64FrameRequest):
    try:
        data_str = req.image_base64
        if "," in data_str:
            data_str = data_str.split(",")[1]

        image_bytes = base64.b64decode(data_str)
        temp_cam_file = TEMP_DIR / "webcam_capture.png"
        
        with open(temp_cam_file, "wb") as f:
            f.write(image_bytes)

        result = pipeline.process_file(temp_cam_file, synthesize_audio=True)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/voice_command/{cmd_name}")
def voice_command(cmd_name: str):
    if cmd_name == "read_this":
        return pipeline.read_first_chunk(play_speaker=False)
    elif cmd_name == "continue":
        return pipeline.read_next_chunk(play_speaker=False)
    else:
        raise HTTPException(status_code=400, detail="Invalid voice command")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
