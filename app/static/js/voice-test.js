document.addEventListener("DOMContentLoaded", () => {
    const voiceForm = document.getElementById("voiceForm");
    if (!voiceForm) {
        return; // Not on the voice test page, exit.
    }

    // --- Get Elements ---
    const statusText = document.getElementById("statusText");
    const sessionId = document.getElementById("sessionId").value;
    const uploadUrl = voiceForm.dataset.uploadUrl;

    const tabRecord = document.getElementById("tab-record");
    const tabUpload = document.getElementById("tab-upload");
    const panelRecord = document.getElementById("panel-record");
    const panelUpload = document.getElementById("panel-upload");

    const audioFileInput = document.getElementById("audioFile");
    const dropZone = document.getElementById("drop-zone");
    const fileNameDisplay = document.getElementById("file-name-display");
    const uploadButton = document.getElementById("uploadButton");

    const startButton = document.getElementById("startButton");
    const countdownText = document.getElementById("countdownText");

    // --- Tab Switching Logic ---
    tabRecord.addEventListener("click", () => {
        panelRecord.classList.remove("hidden");
        panelUpload.classList.add("hidden");
        tabRecord.classList.add("border-teal-500", "text-teal-600");
        tabRecord.classList.remove("border-transparent", "text-slate-500");
        tabUpload.classList.add("border-transparent", "text-slate-500");
        tabUpload.classList.remove("border-teal-500", "text-teal-600");
    });

    tabUpload.addEventListener("click", () => {
        panelUpload.classList.remove("hidden");
        panelRecord.classList.add("hidden");
        tabUpload.classList.add("border-teal-500", "text-teal-600");
        tabUpload.classList.remove("border-transparent", "text-slate-500");
        tabRecord.classList.add("border-transparent", "text-slate-500");
        tabRecord.classList.remove("border-teal-500", "text-teal-600");
    });

    // --- File Upload Panel Logic ---
    dropZone.addEventListener("click", () => audioFileInput.click());
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("border-teal-500", "bg-teal-50");
    });
    dropZone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dropZone.classList.remove("border-teal-500", "bg-teal-50");
    });
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("border-teal-500", "bg-teal-50");
        if (e.dataTransfer.files.length > 0 && e.dataTransfer.files[0].type === "audio/wav") {
            audioFileInput.files = e.dataTransfer.files;
            updateFileName();
        }
    });
    audioFileInput.addEventListener("change", updateFileName);

    function updateFileName() {
        if (audioFileInput.files.length > 0) {
            fileNameDisplay.textContent = audioFileInput.files[0].name;
            uploadButton.disabled = false;
        } else {
            fileNameDisplay.textContent = "WAV up to 10MB";
            uploadButton.disabled = true;
        }
    }

    uploadButton.addEventListener("click", () => {
        if (audioFileInput.files.length > 0) {
            const formData = new FormData();
            formData.append("session_id", sessionId);
            formData.append("file", audioFileInput.files[0]);
            submitFormData(formData, uploadButton, "Submit Uploaded File");
        }
    });

    // --- Browser Recording Logic ---
    let audioContext, mediaStream, processorNode, sourceNode, silenceNode;
    let audioChunks = [];
    let countdownTimer = null;

    function writeString(view, offset, str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }

    function encodeWav(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        const numChannels = 1;
        const bitsPerSample = 16;
        const blockAlign = numChannels * (bitsPerSample / 8);
        const byteRate = sampleRate * blockAlign;

        writeString(view, 0, "RIFF");
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, "WAVE");
        writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        writeString(view, 36, "data");
        view.setUint32(40, samples.length * 2, true);
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
        return new Blob([view], { type: "audio/wav" });
    }

    function mergeBuffers(buffers) {
        const totalLength = buffers.reduce((acc, b) => acc + b.length, 0);
        const merged = new Float32Array(totalLength);
        let offset = 0;
        buffers.forEach((buffer) => {
            merged.set(buffer, offset);
            offset += buffer.length;
        });
        return merged;
    }

    async function stopAndUpload() {
        if (countdownTimer) {
            clearInterval(countdownTimer);
            countdownTimer = null;
        }
        countdownText.textContent = "0";
        if (!audioContext || !mediaStream) {
            statusText.textContent = "No recording captured.";
            startButton.disabled = false;
            return;
        }

        processorNode?.disconnect();
        sourceNode?.disconnect();
        silenceNode?.disconnect();
        mediaStream.getTracks().forEach((track) => track.stop());

        const samples = mergeBuffers(audioChunks);
        const wavBlob = encodeWav(samples, audioContext.sampleRate);
        if (audioContext.state !== "closed") {
            await audioContext.close();
        }

        const formData = new FormData();
        formData.append("session_id", sessionId);
        formData.append("file", wavBlob, "recording.wav");
        submitFormData(formData, startButton, "Start Microphone");
    }

    async function startRecording() {
        try {
            statusText.textContent = "Requesting microphone permission...";
            startButton.disabled = true;
            startButton.textContent = "Listening...";
            let countdownSeconds = 5;
            countdownText.textContent = String(countdownSeconds);

            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            sourceNode = audioContext.createMediaStreamSource(mediaStream);
            processorNode = audioContext.createScriptProcessor(4096, 1, 1);
            silenceNode = audioContext.createGain();
            silenceNode.gain.value = 0;
            audioChunks = [];

            processorNode.onaudioprocess = (e) => {
                audioChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
            };
            sourceNode.connect(processorNode);
            processorNode.connect(silenceNode);
            silenceNode.connect(audioContext.destination);

            statusText.textContent = "Recording. Please speak clearly.";
            countdownTimer = setInterval(() => {
                countdownSeconds -= 1;
                countdownText.textContent = String(Math.max(countdownSeconds, 0));
                if (countdownSeconds <= 0) {
                    stopAndUpload();
                }
            }, 1000);
        } catch (error) {
            console.error("Error during recording:", error);
            statusText.textContent = "Microphone permission was not granted or an error occurred.";
            startButton.disabled = false;
            startButton.textContent = "Start Microphone";
        }
    }
    startButton.addEventListener("click", startRecording);

    // --- Shared Form Submission Logic ---
    async function submitFormData(formData, buttonElement, buttonText) {
        statusText.textContent = "Uploading and processing...";
        buttonElement.disabled = true;
        buttonElement.textContent = "Processing...";

        try {
            const response = await fetch(uploadUrl, {
                method: "POST",
                body: formData,
                headers: { "X-Requested-With": "XMLHttpRequest" },
            });
            const payload = await response.json();
            if (!response.ok || !payload.success) {
                throw new Error(payload.message || "The sample could not be processed.");
            }
            window.location.href = payload.redirect_url;
        } catch (error) {
            console.error("Error submitting form:", error);
            statusText.textContent = `Error: ${error.message}`;
            buttonElement.disabled = false;
            buttonElement.textContent = buttonText;
        }
    }
});
