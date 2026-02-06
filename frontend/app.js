/**
 * Kaleidoscope Studio - Frontend Application
 * Audio-reactive visualization controller
 */

class KaleidoscopeStudio {
    constructor() {
        // Configuration state
        this.config = {
            // Geometry
            mirrors: 8,
            baseRadius: 150,
            orbitRadius: 200,
            rotationSpeed: 2.0,
            // Dynamics
            maxScale: 1.8,
            trailAlpha: 40,
            attackMs: 0,
            releaseMs: 200,
            // Shape
            minSides: 3,
            maxSides: 12,
            baseThickness: 3,
            maxThickness: 12,
            // Colors
            bgColor: '#05050f',
            accentColor: '#f59e0b',
            chromaColors: true,
            saturation: 85,
            // Export
            width: 1920,
            height: 1080,
            fps: 60
        };

        // Audio state
        this.audioContext = null;
        this.audioSource = null;
        this.analyser = null;
        this.audioBuffer = null;
        this.isPlaying = false;
        this.startTime = 0;
        this.pauseTime = 0;
        this.duration = 0;
        this.manifest = null;
        this.currentFrame = 0;

        // Visualization state
        this.accumulatedRotation = 0;
        this.lastFrameTime = 0;

        // Canvas
        this.canvas = document.getElementById('visualizerCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.waveformCanvas = document.getElementById('waveformCanvas');
        this.waveformCtx = this.waveformCanvas.getContext('2d');

        // Chroma to hue mapping
        this.chromaToHue = {
            'C': 0, 'C#': 30, 'D': 60, 'D#': 90, 'E': 120, 'F': 150,
            'F#': 180, 'G': 210, 'G#': 240, 'A': 270, 'A#': 300, 'B': 330
        };

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupKnobs();
        this.render();
        this.startAnimationLoop();
    }

    setupEventListeners() {
        // Audio upload
        const uploadZone = document.getElementById('uploadZone');
        const audioInput = document.getElementById('audioInput');

        uploadZone.addEventListener('click', () => audioInput.click());
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                this.loadAudioFile(e.dataTransfer.files[0]);
            }
        });
        audioInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                this.loadAudioFile(e.target.files[0]);
            }
        });

        // Transport controls
        document.getElementById('playBtn').addEventListener('click', () => this.togglePlay());
        document.getElementById('skipBackBtn').addEventListener('click', () => this.skip(-10));
        document.getElementById('skipForwardBtn').addEventListener('click', () => this.skip(10));

        // Volume
        document.getElementById('volumeSlider').addEventListener('input', (e) => {
            if (this.gainNode) {
                this.gainNode.gain.value = e.target.value / 100;
            }
        });

        // Waveform click to seek
        const waveformContainer = document.getElementById('waveformContainer');
        waveformContainer.addEventListener('click', (e) => {
            if (!this.duration) return;
            const rect = waveformContainer.getBoundingClientRect();
            const ratio = (e.clientX - rect.left) / rect.width;
            this.seekTo(ratio * this.duration);
        });

        // Sliders
        document.getElementById('attackSlider').addEventListener('input', (e) => {
            this.config.attackMs = parseInt(e.target.value);
            document.getElementById('attackValue').textContent = `${e.target.value} ms`;
        });

        document.getElementById('releaseSlider').addEventListener('input', (e) => {
            this.config.releaseMs = parseInt(e.target.value);
            document.getElementById('releaseValue').textContent = `${e.target.value} ms`;
        });

        document.getElementById('saturationSlider').addEventListener('input', (e) => {
            this.config.saturation = parseInt(e.target.value);
            document.getElementById('saturationValue').textContent = `${e.target.value}%`;
        });

        // Colors
        document.getElementById('bgColor').addEventListener('input', (e) => {
            this.config.bgColor = e.target.value;
        });

        document.getElementById('accentColor').addEventListener('input', (e) => {
            this.config.accentColor = e.target.value;
        });

        document.getElementById('chromaColors').addEventListener('change', (e) => {
            this.config.chromaColors = e.target.checked;
        });

        // Resolution buttons
        document.querySelectorAll('[data-resolution]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-resolution]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                const [w, h] = e.target.dataset.resolution.split('x').map(Number);
                this.config.width = w;
                this.config.height = h;
                document.getElementById('resolutionBadge').textContent = `${w} × ${h}`;
            });
        });

        // FPS buttons
        document.querySelectorAll('[data-fps]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-fps]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.config.fps = parseInt(e.target.dataset.fps);
                document.getElementById('fpsBadge').textContent = `${this.config.fps} FPS`;
            });
        });

        // Export button
        document.getElementById('exportBtn').addEventListener('click', () => this.exportVideo());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !e.target.matches('input')) {
                e.preventDefault();
                this.togglePlay();
            }
        });
    }

    setupKnobs() {
        document.querySelectorAll('.knob').forEach(knob => {
            const param = knob.dataset.param;
            const min = parseFloat(knob.dataset.min);
            const max = parseFloat(knob.dataset.max);
            const step = parseFloat(knob.dataset.step) || 1;
            let value = parseFloat(knob.dataset.value);

            const updateKnob = (newValue) => {
                value = Math.max(min, Math.min(max, newValue));
                this.config[param] = value;

                // Update visual rotation (270 degree range)
                const ratio = (value - min) / (max - min);
                const angle = -135 + (ratio * 270);
                knob.querySelector('.knob-indicator').style.transform = `rotate(${angle}deg)`;

                // Update value display
                const display = step < 1 ? value.toFixed(1) : Math.round(value);
                document.getElementById(`${param}Value`).textContent = display;
            };

            // Initialize
            updateKnob(value);

            // Drag handling
            let startY, startValue;

            const onMouseDown = (e) => {
                e.preventDefault();
                startY = e.clientY;
                startValue = value;
                knob.classList.add('active');
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            };

            const onMouseMove = (e) => {
                const deltaY = startY - e.clientY;
                const range = max - min;
                const sensitivity = range / 150; // pixels per full range
                const newValue = startValue + (deltaY * sensitivity);
                const snapped = Math.round(newValue / step) * step;
                updateKnob(snapped);
            };

            const onMouseUp = () => {
                knob.classList.remove('active');
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
            };

            knob.addEventListener('mousedown', onMouseDown);

            // Mouse wheel
            knob.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? -step : step;
                updateKnob(value + delta);
            });
        });
    }

    async loadAudioFile(file) {
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = statusIndicator.querySelector('.status-text');
        statusIndicator.classList.add('processing');
        statusText.textContent = 'Loading...';

        try {
            // Initialize audio context if needed
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                this.gainNode = this.audioContext.createGain();
                this.gainNode.connect(this.audioContext.destination);
                this.gainNode.gain.value = 0.8;
            }

            // Decode audio
            const arrayBuffer = await file.arrayBuffer();
            this.audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            this.duration = this.audioBuffer.duration;

            // Update UI
            document.getElementById('trackInfo').style.display = 'block';
            document.getElementById('trackName').textContent = file.name.replace(/\.[^/.]+$/, '');
            document.getElementById('trackDuration').textContent = this.formatTime(this.duration);
            document.getElementById('totalTime').textContent = this.formatTime(this.duration);
            document.getElementById('canvasOverlay').classList.add('hidden');
            document.getElementById('uploadZone').style.display = 'none';

            // Draw waveform
            this.drawWaveform();

            // Analyze audio (simulate manifest data for now)
            statusText.textContent = 'Analyzing...';
            await this.analyzeAudio(file);

            statusIndicator.classList.remove('processing');
            statusText.textContent = 'Ready';

        } catch (error) {
            console.error('Error loading audio:', error);
            statusIndicator.classList.remove('processing');
            statusIndicator.classList.add('error');
            statusText.textContent = 'Error';
        }
    }

    async analyzeAudio(file) {
        // For now, generate simulated manifest data
        // In production, this would call the Python backend
        const fps = this.config.fps;
        const totalFrames = Math.ceil(this.duration * fps);
        const frames = [];

        // Create simulated beat pattern (120 BPM default)
        const beatsPerSecond = 2; // 120 BPM
        const samplesPerBeat = fps / beatsPerSecond;

        for (let i = 0; i < totalFrames; i++) {
            const time = i / fps;
            const beatPhase = (i % samplesPerBeat) / samplesPerBeat;
            const isBeat = beatPhase < 0.05;

            // Simulate energy with some variation
            const baseEnergy = 0.5 + 0.3 * Math.sin(time * 0.5);
            const percussiveImpact = isBeat ? 0.8 + Math.random() * 0.2 : 0.1 + Math.random() * 0.2;
            const harmonicEnergy = baseEnergy + 0.2 * Math.sin(time * 2);

            // Cycle through chroma
            const chromaNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
            const chromaIndex = Math.floor(time / 4) % 12;

            frames.push({
                frame_index: i,
                time: time,
                is_beat: isBeat,
                is_onset: isBeat,
                percussive_impact: Math.min(1, percussiveImpact),
                harmonic_energy: Math.min(1, harmonicEnergy),
                global_energy: (percussiveImpact + harmonicEnergy) / 2,
                spectral_brightness: 0.5 + 0.3 * Math.sin(time * 3),
                dominant_chroma: chromaNames[chromaIndex]
            });
        }

        this.manifest = {
            metadata: {
                bpm: 120,
                duration: this.duration,
                fps: fps,
                n_frames: totalFrames
            },
            frames: frames
        };

        document.getElementById('trackBpm').textContent = `${this.manifest.metadata.bpm} BPM`;
    }

    drawWaveform() {
        const canvas = this.waveformCanvas;
        const ctx = this.waveformCtx;
        const data = this.audioBuffer.getChannelData(0);

        // Resize canvas
        canvas.width = canvas.offsetWidth * 2;
        canvas.height = canvas.offsetHeight * 2;
        ctx.scale(2, 2);

        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        const step = Math.ceil(data.length / width);
        const amp = height / 2;

        ctx.fillStyle = '#1a1a24';
        ctx.fillRect(0, 0, width, height);

        ctx.beginPath();
        ctx.moveTo(0, amp);

        for (let i = 0; i < width; i++) {
            let min = 1.0;
            let max = -1.0;
            for (let j = 0; j < step; j++) {
                const idx = (i * step) + j;
                if (idx < data.length) {
                    const datum = data[idx];
                    if (datum < min) min = datum;
                    if (datum > max) max = datum;
                }
            }

            // Draw filled waveform
            const y1 = (1 + min) * amp;
            const y2 = (1 + max) * amp;
            ctx.lineTo(i, y1);
        }

        // Complete the shape going backwards
        for (let i = width - 1; i >= 0; i--) {
            let max = -1.0;
            for (let j = 0; j < step; j++) {
                const idx = (i * step) + j;
                if (idx < data.length) {
                    const datum = data[idx];
                    if (datum > max) max = datum;
                }
            }
            const y2 = (1 + max) * amp;
            ctx.lineTo(i, y2);
        }

        ctx.closePath();

        // Gradient fill
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, 'rgba(245, 158, 11, 0.6)');
        gradient.addColorStop(0.5, 'rgba(245, 158, 11, 0.3)');
        gradient.addColorStop(1, 'rgba(245, 158, 11, 0.6)');
        ctx.fillStyle = gradient;
        ctx.fill();
    }

    togglePlay() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    play() {
        if (!this.audioBuffer) return;

        if (this.audioSource) {
            this.audioSource.stop();
        }

        this.audioSource = this.audioContext.createBufferSource();
        this.audioSource.buffer = this.audioBuffer;
        this.audioSource.connect(this.gainNode);

        const offset = this.pauseTime || 0;
        this.startTime = this.audioContext.currentTime - offset;
        this.audioSource.start(0, offset);

        this.isPlaying = true;
        document.getElementById('playBtn').classList.add('playing');

        this.audioSource.onended = () => {
            if (this.isPlaying) {
                this.stop();
            }
        };
    }

    pause() {
        if (!this.isPlaying) return;

        this.audioSource.stop();
        this.pauseTime = this.audioContext.currentTime - this.startTime;
        this.isPlaying = false;
        document.getElementById('playBtn').classList.remove('playing');
    }

    stop() {
        if (this.audioSource) {
            this.audioSource.stop();
        }
        this.isPlaying = false;
        this.pauseTime = 0;
        document.getElementById('playBtn').classList.remove('playing');
    }

    skip(seconds) {
        if (!this.duration) return;
        const currentTime = this.isPlaying
            ? this.audioContext.currentTime - this.startTime
            : this.pauseTime;
        this.seekTo(Math.max(0, Math.min(this.duration, currentTime + seconds)));
    }

    seekTo(time) {
        const wasPlaying = this.isPlaying;
        if (wasPlaying) {
            this.audioSource.stop();
        }
        this.pauseTime = time;

        if (wasPlaying) {
            this.play();
        }
    }

    getCurrentTime() {
        if (!this.audioBuffer) return 0;
        if (this.isPlaying) {
            return Math.min(this.audioContext.currentTime - this.startTime, this.duration);
        }
        return this.pauseTime || 0;
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    startAnimationLoop() {
        const animate = (timestamp) => {
            const currentTime = this.getCurrentTime();

            // Update time display
            document.getElementById('currentTime').textContent = this.formatTime(currentTime);

            // Update playhead
            if (this.duration > 0) {
                const ratio = currentTime / this.duration;
                document.getElementById('playhead').style.left = `${ratio * 100}%`;
            }

            // Get current frame data
            let frameData = null;
            if (this.manifest && this.manifest.frames) {
                const frameIndex = Math.floor(currentTime * this.config.fps);
                frameData = this.manifest.frames[Math.min(frameIndex, this.manifest.frames.length - 1)];

                // Beat flash effect
                if (frameData && frameData.is_beat && this.isPlaying) {
                    document.querySelector('.canvas-container').classList.add('beat');
                    setTimeout(() => {
                        document.querySelector('.canvas-container').classList.remove('beat');
                    }, 100);
                }
            }

            // Render frame
            this.renderFrame(frameData, timestamp);

            requestAnimationFrame(animate);
        };

        requestAnimationFrame(animate);
    }

    renderFrame(frameData, timestamp) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const config = this.config;

        // Default frame data if none provided
        if (!frameData) {
            frameData = {
                percussive_impact: 0.1,
                harmonic_energy: 0.3,
                is_beat: false,
                spectral_brightness: 0.5,
                dominant_chroma: 'C'
            };
        }

        // Trail effect - fade previous frame
        const trailAlpha = config.trailAlpha / 255;
        ctx.fillStyle = config.bgColor;
        ctx.globalAlpha = 1 - trailAlpha;
        ctx.fillRect(0, 0, width, height);
        ctx.globalAlpha = 1;

        const centerX = width / 2;
        const centerY = height / 2;

        // Calculate visual parameters from audio
        const scale = 1 + (frameData.percussive_impact * (config.maxScale - 1));
        const radius = config.baseRadius * scale;

        // Rotation accumulation
        const deltaTime = timestamp - this.lastFrameTime;
        this.lastFrameTime = timestamp;
        if (this.isPlaying) {
            this.accumulatedRotation += frameData.harmonic_energy * config.rotationSpeed * (deltaTime / 1000);
        }

        // Polygon sides based on brightness
        const numSides = Math.round(
            config.minSides + frameData.spectral_brightness * (config.maxSides - config.minSides)
        );

        // Thickness based on percussive impact
        const thickness = config.baseThickness +
            frameData.percussive_impact * (config.maxThickness - config.baseThickness);

        // Get color
        let hue;
        if (config.chromaColors) {
            hue = this.chromaToHue[frameData.dominant_chroma] || 0;
        } else {
            hue = this.hexToHsl(config.accentColor).h;
        }

        const orbitDistance = config.orbitRadius * (0.5 + frameData.harmonic_energy * 0.5);

        // Draw kaleidoscope pattern
        for (let i = 0; i < config.mirrors; i++) {
            const mirrorAngle = (Math.PI * 2 * i / config.mirrors) + this.accumulatedRotation * 0.3;

            const orbitX = centerX + orbitDistance * Math.cos(mirrorAngle);
            const orbitY = centerY + orbitDistance * Math.sin(mirrorAngle);

            // Outer polygon
            this.drawPolygon(
                ctx,
                orbitX, orbitY,
                radius * 0.8,
                numSides,
                this.accumulatedRotation + mirrorAngle,
                `hsl(${hue}, ${config.saturation}%, 70%)`,
                thickness
            );

            // Inner polygon (counter-rotating)
            const innerHue = (hue + 180) % 360;
            this.drawPolygon(
                ctx,
                orbitX, orbitY,
                radius * 0.4,
                Math.max(3, numSides - 2),
                -this.accumulatedRotation * 1.5 + mirrorAngle,
                `hsl(${innerHue}, ${config.saturation * 0.8}%, 60%)`,
                Math.max(1, thickness / 2)
            );
        }

        // Central polygon
        this.drawPolygon(
            ctx,
            centerX, centerY,
            radius * 0.6,
            numSides,
            this.accumulatedRotation * 0.5,
            `hsl(${hue}, ${config.saturation}%, 80%)`,
            thickness + 2
        );
    }

    drawPolygon(ctx, x, y, radius, sides, rotation, color, thickness) {
        if (sides < 3) return;

        ctx.beginPath();
        for (let i = 0; i < sides; i++) {
            const angle = rotation + (Math.PI * 2 * i / sides);
            const px = x + radius * Math.cos(angle);
            const py = y + radius * Math.sin(angle);
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.closePath();

        ctx.strokeStyle = color;
        ctx.lineWidth = thickness;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();
    }

    hexToHsl(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        if (!result) return { h: 0, s: 50, l: 50 };

        let r = parseInt(result[1], 16) / 255;
        let g = parseInt(result[2], 16) / 255;
        let b = parseInt(result[3], 16) / 255;

        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;

        if (max === min) {
            h = s = 0;
        } else {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch (max) {
                case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
                case g: h = ((b - r) / d + 2) / 6; break;
                case b: h = ((r - g) / d + 4) / 6; break;
            }
        }

        return { h: h * 360, s: s * 100, l: l * 100 };
    }

    render() {
        // Initial render with idle state
        this.renderFrame(null, 0);
    }

    async exportVideo() {
        if (!this.manifest) {
            alert('Please load an audio file first');
            return;
        }

        const exportBtn = document.getElementById('exportBtn');
        const exportProgress = document.getElementById('exportProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');

        exportBtn.disabled = true;
        exportProgress.style.display = 'block';

        try {
            // Get the audio file
            const audioInput = document.getElementById('audioInput');
            if (!audioInput.files.length) {
                throw new Error('No audio file loaded');
            }

            const formData = new FormData();
            formData.append('audio', audioInput.files[0]);
            formData.append('config', JSON.stringify(this.config));

            // Send to backend for rendering
            const response = await fetch('/api/render', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Export failed');
            }

            // Poll for progress
            const taskId = (await response.json()).task_id;
            await this.pollExportProgress(taskId, progressFill, progressText);

        } catch (error) {
            console.error('Export error:', error);
            progressText.textContent = 'Export failed: ' + error.message;

            // Fallback: Show message about using CLI
            setTimeout(() => {
                alert(
                    'To export video, use the command line:\n\n' +
                    'python -m audio_analysisussy.render_video your_audio.mp3\n\n' +
                    'The frontend preview uses the same visualization engine.'
                );
            }, 1000);
        } finally {
            setTimeout(() => {
                exportBtn.disabled = false;
                exportProgress.style.display = 'none';
                progressFill.style.width = '0%';
            }, 3000);
        }
    }

    async pollExportProgress(taskId, progressFill, progressText) {
        while (true) {
            const response = await fetch(`/api/render/status/${taskId}`);
            const status = await response.json();

            progressFill.style.width = `${status.progress}%`;
            progressText.textContent = status.message;

            if (status.complete) {
                if (status.output_path) {
                    // Trigger download
                    window.location.href = `/api/render/download/${taskId}`;
                }
                break;
            }

            if (status.error) {
                throw new Error(status.error);
            }

            await new Promise(r => setTimeout(r, 500));
        }
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.studio = new KaleidoscopeStudio();
});
