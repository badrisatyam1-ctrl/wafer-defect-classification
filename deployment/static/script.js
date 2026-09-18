/**
 * WaferOS Client Controller
 * Powers Apple-aesthetic Wafer Yield Analytics Studio
 */

// ── State Management ──────────────────────────────────────────────────
const DEFECT_COLORS = {
    "normal": "#30d158",
    "center": "#2997ff",
    "edge_ring": "#bf5af2",
    "edge_loss": "#ff9f0a",
    "scratch": "#ff453a",
    "ring": "#64d2ff",
    "cluster": "#ff375f",
    "full_fail": "#d70015"
};

let currentView = 'classifier';
let currentMode = 'upload';
let selectedSyntheticClass = 'scratch';
let activeFile = null;
let activeImageB64 = null;
let currentResult = null;
let sessionHistory = [];
let webcamStream = null;

// Chart references
let defectChart = null;
let confidenceChart = null;
let yieldTrendChart = null;

// ── Initialization ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    loadSystemInfo();
    loadSampleWafer(); // Load initial sample so the interface is immediately engaging
});

// ── View Switching ───────────────────────────────────────────────────
function switchView(viewName) {
    currentView = viewName;

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewName);
    });

    // Update view containers
    document.querySelectorAll('.app-view').forEach(view => {
        view.classList.toggle('active', view.id === `view-${viewName}`);
    });

    // View-specific actions
    if (viewName === 'analytics') {
        renderAnalyticsCharts();
    } else if (viewName === 'history') {
        renderHistoryTable();
    }
}

// ── Classifier Input Modes ───────────────────────────────────────────
function setClassifierMode(mode) {
    currentMode = mode;

    document.querySelectorAll('.mode-pill').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    document.querySelectorAll('.mode-content').forEach(el => {
        el.classList.toggle('active', el.id === `mode-${mode}`);
    });

    // Stop webcam if switching away from camera mode
    if (mode !== 'camera' && webcamStream) {
        stopCamera();
    }
}

function selectSyntheticClass(className) {
    selectedSyntheticClass = className;
    document.querySelectorAll('.class-chip').forEach(chip => {
        chip.classList.toggle('active', chip.dataset.class === className);
    });
}

// ── Dropzone & File Handling ─────────────────────────────────────────
function initDropzone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('wafer-file-input');

    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });
}

function handleSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please select a valid image file (PNG, JPG, BMP).');
        return;
    }

    activeFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        activeImageB64 = e.target.result;
        showSourcePreview(file.name, activeImageB64);
        // Automatically run classification
        executeClassification();
    };
    reader.readAsDataURL(file);
}

function showSourcePreview(filename, src) {
    const container = document.getElementById('source-preview-container');
    const img = document.getElementById('source-preview-img');
    const label = document.getElementById('source-filename');

    img.src = src;
    label.textContent = filename;
    container.classList.remove('hidden');
}

function clearActiveImage() {
    activeFile = null;
    activeImageB64 = null;
    document.getElementById('source-preview-container').classList.add('hidden');
    document.getElementById('wafer-file-input').value = '';
}

// ── Synthetic Wafer Generation ───────────────────────────────────────
async function generateSyntheticWafer() {
    const btn = document.getElementById('btn-generate-synthetic');
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-inline"></span> Generating ${selectedSyntheticClass.replace('_', ' ')}...`;

    showDiagnosticsLoading();

    try {
        const response = await fetch('/api/synthetic', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ defect_class: selectedSyntheticClass })
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${await response.text()}`);
        }

        const data = await response.json();
        activeImageB64 = data.input_b64;
        showSourcePreview(`synthetic_${selectedSyntheticClass}.png`, activeImageB64);
        renderInspectionResult(data);
        showToast(`Synthesized ${data.class.replace('_', ' ').toUpperCase()} wafer defect.`);

    } catch (err) {
        console.error('Synthetic error:', err);
        showToast(`Failed to generate: ${err.message}`);
        hideDiagnosticsLoading();
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/></svg> Synthesize &amp; Classify`;
    }
}

// ── Realtime Camera Capture ──────────────────────────────────────────
async function startCamera() {
    const video = document.getElementById('webcam-video');
    const snapBtn = document.getElementById('btn-snap-camera');
    const startBtn = document.getElementById('btn-start-camera');

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 640 } }
        });
        video.srcObject = webcamStream;
        snapBtn.disabled = false;
        startBtn.textContent = 'Stop Camera';
        startBtn.onclick = stopCamera;
        showToast('Camera feed connected.');
    } catch (err) {
        console.error('Camera access error:', err);
        showToast('Camera access denied or unavailable.');
    }
}

function stopCamera() {
    const video = document.getElementById('webcam-video');
    const snapBtn = document.getElementById('btn-snap-camera');
    const startBtn = document.getElementById('btn-start-camera');

    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
    video.srcObject = null;
    snapBtn.disabled = true;
    startBtn.textContent = 'Start Camera';
    startBtn.onclick = startCamera;
}

function captureCamera() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    if (!video.videoWidth) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
        const file = new File([blob], `camera_wafer_${Date.now()}.png`, { type: 'image/png' });
        handleSelectedFile(file);
    }, 'image/png');
}

// ── Execute Classification API ───────────────────────────────────────
async function executeClassification(force = false) {
    if (!activeFile && !activeImageB64) {
        showToast('Please select or generate a wafer image first.');
        return;
    }

    showDiagnosticsLoading();

    try {
        let response;
        const endpoint = force ? '/api/predict?force=true' : '/api/predict';
        if (activeFile) {
            const formData = new FormData();
            formData.append('file', activeFile);
            response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
        } else {
            // Convert base64 data to blob
            const res = await fetch(activeImageB64);
            const blob = await res.blob();
            const formData = new FormData();
            formData.append('file', blob, 'wafer.png');
            response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
        }

        if (!response.ok) {
            throw new Error(`Inference error: ${response.status} ${await response.text()}`);
        }

        const data = await response.json();
        if (data.status === 'rejected') {
            renderRejectionResult(data);
            return;
        }

        renderInspectionResult(data);

    } catch (err) {
        console.error('Classification error:', err);
        showToast(`Analysis failed: ${err.message}`);
        hideDiagnosticsLoading();
    }
}

// ── Render Rejection (Non-Wafer Filtered) ────────────────────────────
function renderRejectionResult(data) {
    hideDiagnosticsLoading();

    document.getElementById('results-empty-state').classList.add('hidden');
    document.getElementById('results-active-content').classList.add('hidden');
    document.getElementById('results-meta-pill').classList.add('hidden');

    const rejectCard = document.getElementById('results-rejected-state');
    if (rejectCard) {
        rejectCard.classList.remove('hidden');
    }
    const msgEl = document.getElementById('rejected-message');
    if (msgEl && data.message) {
        msgEl.textContent = data.message;
    }

    showToast('Frame rejected: No circular silicon wafer disc detected.');
}

// ── Render Diagnostics Results ───────────────────────────────────────
function renderInspectionResult(data) {
    currentResult = data;
    hideDiagnosticsLoading();

    // Hide rejected state if previously shown
    const rejectCard = document.getElementById('results-rejected-state');
    if (rejectCard) rejectCard.classList.add('hidden');

    // Show result content
    document.getElementById('results-empty-state').classList.add('hidden');
    document.getElementById('results-active-content').classList.remove('hidden');
    document.getElementById('results-meta-pill').classList.remove('hidden');

    const defectClass = data.class;
    const displayName = defectClass.replace('_', ' ').toUpperCase();
    const color = DEFECT_COLORS[defectClass] || '#0071e3';

    // Banner
    const banner = document.getElementById('defect-banner');
    banner.style.borderLeft = `5px solid ${color}`;
    document.getElementById('result-defect-name').textContent = displayName;
    document.getElementById('result-defect-desc').textContent = data.knowledge?.description || 'Macro defect pattern detected on wafer.';
    
    // Severity
    const severityPill = document.getElementById('result-severity-pill');
    severityPill.textContent = `SEVERITY: ${data.severity.toUpperCase()}`;
    if (data.severity === 'None') {
        severityPill.style.background = 'rgba(48, 209, 88, 0.15)';
        severityPill.style.color = '#30d158';
    } else if (data.severity === 'Critical') {
        severityPill.style.background = 'rgba(215, 0, 21, 0.2)';
        severityPill.style.color = '#ff453a';
    } else {
        severityPill.style.background = 'rgba(255, 159, 10, 0.15)';
        severityPill.style.color = '#ff9f0a';
    }

    // Confidence
    const confPercent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('result-confidence-val').textContent = confPercent;
    document.getElementById('kpi-conf').textContent = confPercent;
    document.getElementById('kpi-conf').style.color = color;
    document.getElementById('kpi-margin').textContent = (data.top2_margin * 100).toFixed(1) + '%';
    document.getElementById('kpi-severity').textContent = data.severity;
    document.getElementById('kpi-speed').textContent = `${data.inference_ms}ms`;
    document.getElementById('latency-tag').textContent = `${data.inference_ms}ms`;

    // Grad-CAM images
    const overlayImg = document.getElementById('cam-overlay-img');
    const rawImg = document.getElementById('cam-raw-img');
    if (data.overlay_b64) {
        overlayImg.src = data.overlay_b64;
    }
    if (data.heatmap_raw_b64) {
        rawImg.src = data.heatmap_raw_b64;
    }

    // Probabilities
    renderProbabilityBars(data.all_probs, defectClass);

    // Engineering Recommendations
    renderEngineeringActions(data);

    // Save to session history
    addToHistory(data);

    // Sync context to assistant
    updateAssistantContext(displayName, confPercent);
}

function renderProbabilityBars(probs, topClass) {
    const container = document.getElementById('probability-bars-container');
    container.innerHTML = '';

    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([clsName, prob]) => {
        const percent = (prob * 100).toFixed(1);
        const color = DEFECT_COLORS[clsName] || '#86868b';
        const isTop = clsName === topClass;

        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <div class="prob-row-header">
                <div class="prob-name-group">
                    <span class="prob-dot" style="background: ${color};"></span>
                    <span style="${isTop ? 'font-weight: 700; color: #fff;' : ''}">${clsName.replace('_', ' ').toUpperCase()}</span>
                </div>
                <span class="prob-val">${percent}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width: ${percent}%; background: ${color};"></div>
            </div>
        `;
        container.appendChild(row);
    });
}

function renderEngineeringActions(data) {
    const container = document.getElementById('engineering-action-body');
    const k = data.knowledge;

    if (!k || data.class === 'normal') {
        container.innerHTML = `
            <p><strong>Wafer Integrity Verified:</strong> Normal die pattern distribution. No macro defect signatures observed across center, edge, or ring geometries. Standard Fab lot clearance approved.</p>
        `;
        return;
    }

    let rootList = '';
    if (k.root_causes && k.root_causes.length > 0) {
        rootList = k.root_causes.map(c => `<li>${c}</li>`).join('');
    }

    let solList = '';
    if (k.solutions && k.solutions.length > 0) {
        solList = k.solutions.map(s => `<li>${s}</li>`).join('');
    }

    container.innerHTML = `
        <p><strong>Impact Assessment:</strong> ${k.impact || 'Localized die loss requiring metrology audit.'}</p>
        <div style="margin-top: 8px;">
            <strong>Primary Root Causes:</strong>
            <ul>${rootList}</ul>
        </div>
        <div style="margin-top: 8px;">
            <strong>Recommended Fab Actions:</strong>
            <ul>${solList}</ul>
        </div>
    `;
}

function toggleCamView(mode) {
    document.getElementById('btn-cam-overlay').classList.toggle('active', mode === 'overlay');
    document.getElementById('btn-cam-raw').classList.toggle('active', mode === 'raw');
    document.getElementById('cam-overlay-img').classList.toggle('active', mode === 'overlay');
    document.getElementById('cam-raw-img').classList.toggle('active', mode === 'raw');
}

function showDiagnosticsLoading() {
    document.getElementById('results-empty-state').classList.add('hidden');
    document.getElementById('results-active-content').classList.add('hidden');
    const rejectCard = document.getElementById('results-rejected-state');
    if (rejectCard) rejectCard.classList.add('hidden');
    document.getElementById('results-loading-state').classList.remove('hidden');
}

function hideDiagnosticsLoading() {
    document.getElementById('results-loading-state').classList.add('hidden');
}

// ── Sample Wafer Loader ──────────────────────────────────────────────
function loadSampleWafer() {
    selectSyntheticClass('cluster');
    generateSyntheticWafer();
}

// ── History & Audit Management ───────────────────────────────────────
function addToHistory(item) {
    sessionHistory.unshift(item);
    document.getElementById('history-count-badge').textContent = sessionHistory.length;
    updateAnalyticsKPIs();
}

function renderHistoryTable() {
    const tbody = document.getElementById('history-table-body');
    const searchVal = (document.getElementById('history-search-input')?.value || '').toLowerCase();
    const filterClass = document.getElementById('history-defect-filter')?.value || 'all';

    tbody.innerHTML = '';

    if (sessionHistory.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" style="text-align: center; padding: 40px; color: var(--text-secondary);">
                    No inspected wafers in current session. Upload or synthesize a wafer map.
                </td>
            </tr>
        `;
        return;
    }

    const filtered = sessionHistory.filter(item => {
        const matchesSearch = item.filename.toLowerCase().includes(searchVal) || item.class.toLowerCase().includes(searchVal);
        const matchesClass = filterClass === 'all' || item.class === filterClass;
        return matchesSearch && matchesClass;
    });

    filtered.forEach((item, index) => {
        const tr = document.createElement('tr');
        const color = DEFECT_COLORS[item.class] || '#0071e3';

        tr.innerHTML = `
            <td>
                <img src="${item.input_b64 || item.overlay_b64}" class="table-thumb" alt="thumb">
            </td>
            <td style="font-weight: 500;">${item.filename}</td>
            <td>
                <span class="badge-table" style="background: ${color}22; color: ${color}; border: 1px solid ${color}44;">
                    ${item.class.replace('_', ' ').toUpperCase()}
                </span>
            </td>
            <td>${(item.confidence * 100).toFixed(1)}%</td>
            <td>${item.severity}</td>
            <td>${item.inference_ms}ms</td>
            <td style="color: var(--text-secondary);">${item.timestamp}</td>
            <td>
                <button class="btn-text-xs" onclick="inspectHistoryItem(${index})">Inspect</button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function filterHistoryTable() {
    renderHistoryTable();
}

function inspectHistoryItem(idx) {
    const item = sessionHistory[idx];
    if (!item) return;
    switchView('classifier');
    renderInspectionResult(item);
    showSourcePreview(item.filename, item.input_b64 || item.overlay_b64);
    showToast(`Loaded ${item.filename}`);
}

function clearHistoryLog() {
    if (confirm('Clear inspection history for this session?')) {
        sessionHistory = [];
        document.getElementById('history-count-badge').textContent = '0';
        renderHistoryTable();
        updateAnalyticsKPIs();
        showToast('History cleared.');
    }
}

// ── PDF & CSV Exports ────────────────────────────────────────────────
async function exportCurrentWaferPdf() {
    if (!currentResult) {
        showToast('No active wafer analysis to export.');
        return;
    }
    showToast('Generating high-resolution PDF report...');
    await downloadPdf([currentResult], `wafer_${currentResult.class}_report`);
}

async function exportLotPdf() {
    if (sessionHistory.length === 0) {
        showToast('No inspection history to export.');
        return;
    }
    showToast(`Compiling full lot report for ${sessionHistory.length} wafers...`);
    await downloadPdf(sessionHistory, `wafer_lot_audit_report`);
}

async function downloadPdf(results, filenamePrefix) {
    try {
        const response = await fetch('/api/export/pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: results, batch_name: filenamePrefix })
        });

        if (!response.ok) {
            throw new Error(`PDF generation failed: ${response.status}`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${filenamePrefix}_${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        showToast('PDF downloaded successfully.');
    } catch (err) {
        console.error('PDF export error:', err);
        showToast(`Failed to export PDF: ${err.message}`);
    }
}

async function exportLotCsv() {
    if (sessionHistory.length === 0) {
        showToast('No inspection history to export.');
        return;
    }

    try {
        const response = await fetch('/api/export/csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: sessionHistory })
        });

        if (!response.ok) {
            throw new Error(`CSV generation failed: ${response.status}`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `wafer_lot_data_${Date.now()}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        showToast('CSV data exported successfully.');
    } catch (err) {
        console.error('CSV export error:', err);
        showToast(`Failed to export CSV: ${err.message}`);
    }
}

function triggerQuickReport() {
    if (sessionHistory.length > 0) {
        exportLotPdf();
    } else if (currentResult) {
        exportCurrentWaferPdf();
    } else {
        showToast('Inspect at least one wafer before exporting reports.');
    }
}

// ── Assistant / Chatbot ──────────────────────────────────────────────
function updateAssistantContext(defectName, confidence) {
    const contextEl = document.getElementById('chat-active-defect');
    if (contextEl) {
        contextEl.textContent = `${defectName} (${confidence})`;
    }
}

function resetChatContext() {
    updateAssistantContext('Any Defect / General Inquiries', 'Fab Mode');
    showToast('Switched to general fab context.');
}

function queryChatbotWithDefect(question) {
    switchView('assistant');
    sendChatQuestion(question);
}

function sendQuickQuery(query) {
    sendChatQuestion(query);
}

function handleChatKey(e) {
    if (e.key === 'Enter') {
        sendChatMessage();
    }
}

function sendChatMessage() {
    const input = document.getElementById('chat-user-input');
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    sendChatQuestion(q);
}

async function sendChatQuestion(question) {
    appendChatMessage('user', question);

    const activeDefect = currentResult?.class || 'scratch';
    const confidence = currentResult?.confidence || 0.95;

    const loaderId = appendChatLoader();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                defect_class: activeDefect,
                confidence: confidence
            })
        });

        removeChatLoader(loaderId);

        if (!response.ok) {
            throw new Error(`Chat API error: ${response.status}`);
        }

        const data = await response.json();
        appendChatMessage('assistant', formatMarkdown(data.reply));

    } catch (err) {
        removeChatLoader(loaderId);
        appendChatMessage('assistant', `<em>Advisory network exception: ${err.message}</em>`);
    }
}

function appendChatMessage(role, htmlContent) {
    const viewport = document.getElementById('chat-messages');
    const msg = document.createElement('div');
    msg.className = `chat-msg msg-${role}`;

    const isUser = role === 'user';
    msg.innerHTML = `
        <div class="msg-avatar">
            ${isUser ? 
                `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>` : 
                `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/><line x1="12" y1="3" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="21"/></svg>`}
        </div>
        <div class="msg-body">
            ${!isUser ? `<div class="msg-sender">Wafer Intelligence Agent</div>` : ''}
            <div class="msg-text">${htmlContent}</div>
            <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
    `;

    viewport.appendChild(msg);
    viewport.scrollTop = viewport.scrollHeight;
}

function appendChatLoader() {
    const viewport = document.getElementById('chat-messages');
    const id = 'loader-' + Date.now();
    const loader = document.createElement('div');
    loader.id = id;
    loader.className = 'chat-msg msg-assistant';
    loader.innerHTML = `
        <div class="msg-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/></svg>
        </div>
        <div class="msg-body">
            <span class="spinner-inline"></span> Consulting semiconductor engineering ontology...
        </div>
    `;
    viewport.appendChild(loader);
    viewport.scrollTop = viewport.scrollHeight;
    return id;
}

function removeChatLoader(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function formatMarkdown(text) {
    if (!text) return '';
    let formatted = text
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/### (.*?)(<br>|$)/g, '<h4 style="color:#fff; margin:6px 0 2px 0;">$1</h4>')
        .replace(/## (.*?)(<br>|$)/g, '<h3 style="color:#fff; margin:8px 0 4px 0;">$1</h3>');
    return formatted;
}

// ── Analytics & Charts ───────────────────────────────────────────────
function updateAnalyticsKPIs() {
    const total = sessionHistory.length;
    const normals = sessionHistory.filter(h => h.class === 'normal').length;
    const defects = total - normals;
    const yieldRate = total > 0 ? ((normals / total) * 100).toFixed(1) : '100.0';
    
    let avgConf = 0;
    if (total > 0) {
        const sum = sessionHistory.reduce((acc, h) => acc + h.confidence, 0);
        avgConf = ((sum / total) * 100).toFixed(1);
    }

    document.getElementById('stats-total').textContent = total;
    document.getElementById('stats-normal').textContent = normals;
    document.getElementById('stats-defects').textContent = defects;
    document.getElementById('stats-yield').textContent = `${yieldRate}%`;
    document.getElementById('stats-avg-conf').textContent = total > 0 ? `${avgConf}%` : '--%';
}

function renderAnalyticsCharts() {
    updateAnalyticsKPIs();

    // 1. Defect Distribution Donut Chart
    const counts = {};
    Object.keys(DEFECT_COLORS).forEach(k => counts[k] = 0);
    sessionHistory.forEach(h => {
        counts[h.class] = (counts[h.class] || 0) + 1;
    });

    const labels = Object.keys(counts).map(k => k.replace('_', ' ').toUpperCase());
    const dataVals = Object.values(counts);
    const bgColors = Object.keys(counts).map(k => DEFECT_COLORS[k]);

    const ctxDonut = document.getElementById('chart-defect-distribution');
    if (ctxDonut) {
        if (defectChart) defectChart.destroy();
        defectChart = new Chart(ctxDonut, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: dataVals,
                    backgroundColor: bgColors,
                    borderWidth: 2,
                    borderColor: '#121215'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { color: '#86868b', font: { family: 'Inter', size: 11 } }
                    }
                }
            }
        });
    }

    // 2. Confidence Distribution Bar Chart
    const bins = [0, 0, 0, 0, 0]; // 0-20, 20-40, 40-60, 60-80, 80-100
    sessionHistory.forEach(h => {
        const binIdx = Math.min(4, Math.floor(h.confidence * 5));
        bins[binIdx]++;
    });

    const ctxConf = document.getElementById('chart-confidence-dist');
    if (ctxConf) {
        if (confidenceChart) confidenceChart.destroy();
        confidenceChart = new Chart(ctxConf, {
            type: 'bar',
            data: {
                labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
                datasets: [{
                    label: 'Wafer Count',
                    data: bins,
                    backgroundColor: '#0071e3',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#86868b' } },
                    y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#86868b' } }
                }
            }
        });
    }

    // 3. Cumulative Yield Trend Line
    const yieldPoints = [];
    let passCount = 0;
    const historyChronological = [...sessionHistory].reverse();
    historyChronological.forEach((h, i) => {
        if (h.class === 'normal') passCount++;
        yieldPoints.push(((passCount / (i + 1)) * 100).toFixed(1));
    });

    const ctxYield = document.getElementById('chart-yield-trend');
    if (ctxYield) {
        if (yieldTrendChart) yieldTrendChart.destroy();
        yieldTrendChart = new Chart(ctxYield, {
            type: 'line',
            data: {
                labels: historyChronological.map((_, i) => `#${i + 1}`),
                datasets: [
                    {
                        label: 'Cumulative Yield %',
                        data: yieldPoints,
                        borderColor: '#30d158',
                        backgroundColor: 'rgba(48, 209, 88, 0.08)',
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: '90% Fab Target Threshold',
                        data: Array(historyChronological.length).fill(90),
                        borderColor: '#ff9f0a',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#86868b' } }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#86868b' } },
                    y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#86868b' } }
                }
            }
        });
    }
}

// ── System Info & Taxonomy ───────────────────────────────────────────
async function loadSystemInfo() {
    try {
        const res = await fetch('/api/system_info');
        if (!res.ok) return;
        const info = await res.json();

        // Populate Defect Taxonomy in Architecture View
        const taxonomyContainer = document.getElementById('taxonomy-grid-container');
        if (taxonomyContainer && info.defect_knowledge) {
            taxonomyContainer.innerHTML = '';
            Object.entries(info.defect_knowledge).forEach(([clsKey, know]) => {
                const card = document.createElement('div');
                card.className = 'taxonomy-card';
                const color = DEFECT_COLORS[clsKey] || '#0071e3';

                card.innerHTML = `
                    <div class="taxonomy-name">
                        <span class="prob-dot" style="background: ${color};"></span>
                        ${clsKey.replace('_', ' ').toUpperCase()}
                    </div>
                    <div class="taxonomy-desc">${know.description}</div>
                    <div class="taxonomy-root">
                        <strong>Root Causes:</strong> ${know.root_causes ? know.root_causes[0] : 'Chamber defect'}
                    </div>
                    <button class="btn btn-text-xs mt-1" style="align-self: flex-start;" onclick="simulateClassTest('${clsKey}')">
                        Synthesize Test Pattern ›
                    </button>
                `;
                taxonomyContainer.appendChild(card);
            });
        }
    } catch (err) {
        console.warn('System info load skipped:', err);
    }
}

function simulateClassTest(clsName) {
    switchView('classifier');
    setClassifierMode('synthetic');
    selectSyntheticClass(clsName);
    generateSyntheticWafer();
}

// ── Toast Notifications ──────────────────────────────────────────────
let toastTimer = null;
function showToast(message) {
    const toast = document.getElementById('apple-toast');
    const text = document.getElementById('toast-message');
    if (!toast || !text) return;

    text.textContent = message;
    toast.classList.add('show');

    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toast.classList.remove('show');
    }, 3200);
}
