/**
 * Kaleidoscope Studio - Frontend Application
 * Audio-reactive visualization controller
 */

class KaleidoscopeStudio {
    constructor() {
        // Configuration state
        this.config = {
            // Style
            style: 'geometric', // geometric, glass, flower, spiral
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
            bgColor2: '#1a0a2e',
            accentColor: '#f59e0b',
            chromaColors: true,
            saturation: 85,
            // Background effects
            dynamicBg: true,
            bgReactivity: 70,
            bgParticles: true,
            bgPulse: true,
            // Export
            width: 1920,
            height: 1080,
            fps: 60
        };

        // Background animation state
        this.bgState = {
            gradientAngle: 0,
            pulseIntensity: 0,
            particles: [],
            noiseOffset: 0
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

        // Real-time audio analysis
        this.frequencyData = null;
        this.timeData = null;
        this.prevEnergy = 0;
        this.energyHistory = [];
        this.beatThreshold = 1.3;
        this.lastBeatTime = 0;
        this.beatCooldown = 150; // ms between beats
        this.dominantChroma = 'C';
        this.chromaSmoothed = new Array(12).fill(0);

        // BPM estimation
        this.beatTimes = [];
        this.estimatedBpm = 0;

        // Visualization state
        this.accumulatedRotation = 0;
        this.lastFrameTime = performance.now();

        // Smoothed values for fluid animation
        this.smoothedValues = {
            percussiveImpact: 0,
            harmonicEnergy: 0.3,
            spectralBrightness: 0.5
        };
        this.smoothingFactor = 0.15; // Lower = smoother

        // Canvas
        this.canvas = document.getElementById('visualizerCanvas');
        this.ctx = this.canvas.getContext('2d', { alpha: false }); // Opaque for performance
        this.waveformCanvas = document.getElementById('waveformCanvas');
        this.waveformCtx = this.waveformCanvas.getContext('2d');

        // Set canvas size from config
        this.canvas.width = this.config.width;
        this.canvas.height = this.config.height;

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
        this.initParticles();
        this.render();
        this.startAnimationLoop();
    }

    initParticles() {
        // Create background particles/stars
        this.bgState.particles = [];
        const count = 80;
        for (let i = 0; i < count; i++) {
            this.bgState.particles.push({
                x: Math.random() * this.config.width,
                y: Math.random() * this.config.height,
                size: Math.random() * 2 + 0.5,
                speed: Math.random() * 0.5 + 0.1,
                angle: Math.random() * Math.PI * 2,
                brightness: Math.random() * 0.5 + 0.3,
                pulse: Math.random() * Math.PI * 2
            });
        }
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

        // Style buttons
        document.querySelectorAll('.style-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.config.style = btn.dataset.style;
            });
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

        document.getElementById('bgColor2').addEventListener('input', (e) => {
            this.config.bgColor2 = e.target.value;
        });

        document.getElementById('accentColor').addEventListener('input', (e) => {
            this.config.accentColor = e.target.value;
        });

        document.getElementById('chromaColors').addEventListener('change', (e) => {
            this.config.chromaColors = e.target.checked;
        });

        // Background effects
        document.getElementById('dynamicBg').addEventListener('change', (e) => {
            this.config.dynamicBg = e.target.checked;
        });

        document.getElementById('bgParticles').addEventListener('change', (e) => {
            this.config.bgParticles = e.target.checked;
        });

        document.getElementById('bgPulse').addEventListener('change', (e) => {
            this.config.bgPulse = e.target.checked;
        });

        document.getElementById('bgReactivitySlider').addEventListener('input', (e) => {
            this.config.bgReactivity = parseInt(e.target.value);
            document.getElementById('bgReactivityValue').textContent = `${e.target.value}%`;
        });

        // Fullscreen toggle
        document.getElementById('fullscreenBtn')?.addEventListener('click', () => {
            this.toggleFullscreen();
        });

        // Resolution buttons
        document.querySelectorAll('[data-resolution]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-resolution]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                const [w, h] = e.target.dataset.resolution.split('x').map(Number);
                this.config.width = w;
                this.config.height = h;
                this.canvas.width = w;
                this.canvas.height = h;
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
            }

            // Create analyser node for real-time frequency analysis
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;

            // Create gain node
            this.gainNode = this.audioContext.createGain();
            this.gainNode.gain.value = 0.8;

            // Connect: source -> analyser -> gain -> destination
            this.analyser.connect(this.gainNode);
            this.gainNode.connect(this.audioContext.destination);

            // Create data arrays for analysis
            this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
            this.timeData = new Uint8Array(this.analyser.fftSize);

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

            // Reset analysis state
            this.energyHistory = [];
            this.prevEnergy = 0;
            this.lastBeatTime = 0;
            this.beatTimes = [];
            this.estimatedBpm = 0;
            document.getElementById('trackBpm').textContent = '-- BPM';

            statusIndicator.classList.remove('processing');
            statusText.textContent = 'Ready';

        } catch (error) {
            console.error('Error loading audio:', error);
            statusIndicator.classList.remove('processing');
            statusIndicator.classList.add('error');
            statusText.textContent = 'Error';
        }
    }

    /**
     * Real-time audio analysis using Web Audio API
     * Called every frame during playback to extract current audio features
     */
    analyzeCurrentAudio() {
        if (!this.analyser || !this.isPlaying) {
            return this.getIdleFrameData();
        }

        // Get current frequency and time domain data
        this.analyser.getByteFrequencyData(this.frequencyData);
        this.analyser.getByteTimeDomainData(this.timeData);

        const sampleRate = this.audioContext.sampleRate;
        const binCount = this.analyser.frequencyBinCount;
        const nyquist = sampleRate / 2;

        // Calculate frequency band energies
        // Low: 20-200Hz, Mid: 200-4000Hz, High: 4000-20000Hz
        const lowEnd = Math.floor(200 / nyquist * binCount);
        const midEnd = Math.floor(4000 / nyquist * binCount);

        let lowSum = 0, midSum = 0, highSum = 0, totalSum = 0;
        let lowCount = 0, midCount = 0, highCount = 0;

        for (let i = 0; i < binCount; i++) {
            const value = this.frequencyData[i] / 255;
            totalSum += value;

            if (i < lowEnd) {
                lowSum += value;
                lowCount++;
            } else if (i < midEnd) {
                midSum += value;
                midCount++;
            } else {
                highSum += value;
                highCount++;
            }
        }

        const lowEnergy = lowCount > 0 ? lowSum / lowCount : 0;
        const midEnergy = midCount > 0 ? midSum / midCount : 0;
        const highEnergy = highCount > 0 ? highSum / highCount : 0;
        const globalEnergy = totalSum / binCount;

        // Calculate RMS from time domain for percussive detection
        let rmsSum = 0;
        for (let i = 0; i < this.timeData.length; i++) {
            const sample = (this.timeData[i] - 128) / 128;
            rmsSum += sample * sample;
        }
        const rms = Math.sqrt(rmsSum / this.timeData.length);

        // Spectral centroid (brightness) - weighted average of frequencies
        let centroidNum = 0, centroidDen = 0;
        for (let i = 0; i < binCount; i++) {
            const freq = i * nyquist / binCount;
            const magnitude = this.frequencyData[i] / 255;
            centroidNum += freq * magnitude;
            centroidDen += magnitude;
        }
        const centroid = centroidDen > 0 ? centroidNum / centroidDen : 0;
        // Normalize to 0-1 range (assuming max useful centroid around 8000Hz)
        const spectralBrightness = Math.min(1, centroid / 8000);

        // Beat detection using energy flux
        const currentEnergy = lowEnergy * 2 + rms; // Weight bass frequencies
        this.energyHistory.push(currentEnergy);
        if (this.energyHistory.length > 30) {
            this.energyHistory.shift();
        }

        const avgEnergy = this.energyHistory.reduce((a, b) => a + b, 0) / this.energyHistory.length;
        const energyFlux = currentEnergy - this.prevEnergy;
        this.prevEnergy = currentEnergy;

        // Detect beat: energy spike above threshold, with cooldown
        const now = performance.now();
        const isBeat = energyFlux > 0.15 &&
                       currentEnergy > avgEnergy * this.beatThreshold &&
                       (now - this.lastBeatTime) > this.beatCooldown;

        if (isBeat) {
            this.lastBeatTime = now;
            this.updateBpmEstimate(now);
        }

        // Percussive impact: combination of bass energy and transient detection
        const percussiveImpact = Math.min(1, (lowEnergy * 0.6 + rms * 0.4) * 1.5);

        // Harmonic energy: mid and high frequencies (melodic content)
        const harmonicEnergy = Math.min(1, (midEnergy * 0.7 + highEnergy * 0.3) * 1.3);

        // Chroma estimation (simplified - uses spectral peaks)
        this.updateChroma();

        return {
            percussive_impact: percussiveImpact,
            harmonic_energy: harmonicEnergy,
            global_energy: globalEnergy,
            low_energy: lowEnergy,
            mid_energy: midEnergy,
            high_energy: highEnergy,
            spectral_brightness: spectralBrightness,
            is_beat: isBeat,
            is_onset: isBeat,
            dominant_chroma: this.dominantChroma
        };
    }

    /**
     * Estimate dominant pitch class from frequency spectrum
     */
    updateChroma() {
        if (!this.frequencyData) return;

        const sampleRate = this.audioContext.sampleRate;
        const binCount = this.analyser.frequencyBinCount;
        const nyquist = sampleRate / 2;

        // Note frequencies for octave 4 (middle octave)
        const noteFreqs = [
            261.63, 277.18, 293.66, 311.13, 329.63, 349.23,
            369.99, 392.00, 415.30, 440.00, 466.16, 493.88
        ]; // C4 to B4

        const chromaEnergy = new Array(12).fill(0);

        // Sum energy at each chroma pitch across octaves
        for (let octave = 2; octave <= 6; octave++) {
            const octaveMultiplier = Math.pow(2, octave - 4);
            for (let note = 0; note < 12; note++) {
                const freq = noteFreqs[note] * octaveMultiplier;
                if (freq < 80 || freq > nyquist) continue;

                const bin = Math.round(freq / nyquist * binCount);
                if (bin >= 0 && bin < binCount) {
                    // Average nearby bins for stability
                    let energy = 0;
                    for (let i = Math.max(0, bin - 1); i <= Math.min(binCount - 1, bin + 1); i++) {
                        energy += this.frequencyData[i] / 255;
                    }
                    chromaEnergy[note] += energy / 3;
                }
            }
        }

        // Smooth chroma values
        for (let i = 0; i < 12; i++) {
            this.chromaSmoothed[i] = this.chromaSmoothed[i] * 0.8 + chromaEnergy[i] * 0.2;
        }

        // Find dominant chroma
        let maxChroma = 0;
        let maxIndex = 0;
        for (let i = 0; i < 12; i++) {
            if (this.chromaSmoothed[i] > maxChroma) {
                maxChroma = this.chromaSmoothed[i];
                maxIndex = i;
            }
        }

        const chromaNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        this.dominantChroma = chromaNames[maxIndex];
    }

    /**
     * Return idle frame data when not playing
     */
    getIdleFrameData() {
        return {
            percussive_impact: 0.1,
            harmonic_energy: 0.3,
            global_energy: 0.2,
            spectral_brightness: 0.5,
            is_beat: false,
            is_onset: false,
            dominant_chroma: 'C'
        };
    }

    /**
     * Estimate BPM from detected beats
     */
    updateBpmEstimate(beatTime) {
        this.beatTimes.push(beatTime);

        // Keep last 16 beats for estimation
        if (this.beatTimes.length > 16) {
            this.beatTimes.shift();
        }

        // Need at least 4 beats to estimate
        if (this.beatTimes.length < 4) {
            return;
        }

        // Calculate intervals between beats
        const intervals = [];
        for (let i = 1; i < this.beatTimes.length; i++) {
            intervals.push(this.beatTimes[i] - this.beatTimes[i - 1]);
        }

        // Filter out outliers (intervals outside reasonable BPM range 60-200)
        const validIntervals = intervals.filter(i => i > 300 && i < 1000);

        if (validIntervals.length < 3) return;

        // Calculate median interval for stability
        validIntervals.sort((a, b) => a - b);
        const medianInterval = validIntervals[Math.floor(validIntervals.length / 2)];

        // Convert to BPM
        const bpm = Math.round(60000 / medianInterval);

        // Only update if reasonable
        if (bpm >= 60 && bpm <= 200) {
            this.estimatedBpm = bpm;
            document.getElementById('trackBpm').textContent = `~${bpm} BPM`;
        }
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

        // Create new buffer source
        this.audioSource = this.audioContext.createBufferSource();
        this.audioSource.buffer = this.audioBuffer;

        // Connect source -> analyser (analyser already connected to gain -> destination)
        this.audioSource.connect(this.analyser);

        const offset = this.pauseTime || 0;
        this.startTime = this.audioContext.currentTime - offset;
        this.audioSource.start(0, offset);

        this.isPlaying = true;
        document.getElementById('playBtn').classList.add('playing');

        // Reset analysis state for fresh playback
        this.energyHistory = [];
        this.prevEnergy = 0;

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
        let lastBeatTime = 0;

        const animate = (timestamp) => {
            const currentTime = this.getCurrentTime();

            // Calculate delta time for smooth animation
            const deltaTime = Math.min(timestamp - this.lastFrameTime, 50); // Cap at 50ms
            this.lastFrameTime = timestamp;

            // Update time display
            document.getElementById('currentTime').textContent = this.formatTime(currentTime);

            // Update playhead
            if (this.duration > 0) {
                const ratio = currentTime / this.duration;
                document.getElementById('playhead').style.left = `${ratio * 100}%`;
            }

            // Get REAL-TIME audio analysis (not pre-computed manifest)
            const frameData = this.analyzeCurrentAudio();

            // Beat flash effect on UI
            if (frameData.is_beat && this.isPlaying) {
                if (timestamp - lastBeatTime > 200) { // Cooldown for UI flash
                    lastBeatTime = timestamp;
                    document.querySelector('.canvas-container').classList.add('beat');
                    setTimeout(() => {
                        document.querySelector('.canvas-container').classList.remove('beat');
                    }, 80);
                }
            }

            // Render frame with delta time for smooth animation
            this.renderFrame(frameData, deltaTime);

            requestAnimationFrame(animate);
        };

        requestAnimationFrame(animate);
    }

    // Smooth interpolation helper
    lerp(current, target, factor) {
        return current + (target - current) * factor;
    }

    renderDynamicBackground(frameData, deltaTime) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const config = this.config;
        const reactivity = config.bgReactivity / 100;

        // Parse colors
        const color1 = this.hexToRgb(config.bgColor);
        const color2 = this.hexToRgb(config.bgColor2);

        // Use a separate, even slower smoothed value just for background color
        // This prevents any connection to beat-related values
        if (this._bgColorBlend === undefined) {
            this._bgColorBlend = 0.3;
        }
        // Extremely slow color blend - practically imperceptible frame to frame
        const targetBlend = 0.3 + this.smoothedValues.harmonicEnergy * reactivity * 0.3;
        this._bgColorBlend = this.lerp(this._bgColorBlend, targetBlend, 0.001);
        const blend = this._bgColorBlend;

        // Fixed center - no movement
        const centerX = width / 2;
        const centerY = height / 2;

        // Stable radius
        const baseRadius = Math.max(width, height) * 0.95;

        // Create simple radial gradient
        const gradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, baseRadius
        );

        // Interpolate colors based on the ultra-slow blend value
        const midColor = {
            r: Math.round(color1.r + (color2.r - color1.r) * blend),
            g: Math.round(color1.g + (color2.g - color1.g) * blend),
            b: Math.round(color1.b + (color2.b - color1.b) * blend)
        };

        gradient.addColorStop(0, `rgb(${midColor.r}, ${midColor.g}, ${midColor.b})`);
        gradient.addColorStop(0.6, `rgb(${Math.round(midColor.r * 0.7 + color1.r * 0.3)}, ${Math.round(midColor.g * 0.7 + color1.g * 0.3)}, ${Math.round(midColor.b * 0.7 + color1.b * 0.3)})`);
        gradient.addColorStop(1, `rgb(${color1.r}, ${color1.g}, ${color1.b})`);

        // Fixed fade amount for consistent trails (not reactive)
        const fadeAmount = (100 - config.trailAlpha) / 100;
        ctx.globalAlpha = fadeAmount * 0.06 + 0.01; // Slightly reduced for smoother trails
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);

        // Static vignette
        const vignetteGradient = ctx.createRadialGradient(
            centerX, centerY, height * 0.35,
            centerX, centerY, Math.max(width, height) * 0.85
        );
        vignetteGradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
        vignetteGradient.addColorStop(1, 'rgba(0, 0, 0, 0.35)');
        ctx.globalAlpha = 0.15;
        ctx.fillStyle = vignetteGradient;
        ctx.fillRect(0, 0, width, height);

        ctx.globalAlpha = 1;

        // Render particles if enabled (these move gently)
        if (config.bgParticles) {
            this.renderParticles(deltaTime);
        }

        // Render pulse rings on beats only
        if (config.bgPulse && this.bgState.pulseIntensity > 0.1) {
            this.renderPulseRings();
        }
    }

    renderParticles(deltaTime) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const config = this.config;

        const reactivity = config.bgReactivity / 100;
        // Use harmonic energy for smooth motion, not percussive
        const energyBoost = 1 + this.smoothedValues.harmonicEnergy * reactivity * 0.3;

        ctx.save();

        this.bgState.particles.forEach(particle => {
            // Update position gently - constant slow drift
            particle.x += Math.cos(particle.angle) * particle.speed * deltaTime * 0.02 * energyBoost;
            particle.y += Math.sin(particle.angle) * particle.speed * deltaTime * 0.02 * energyBoost;

            // Wrap around edges
            if (particle.x < 0) particle.x = width;
            if (particle.x > width) particle.x = 0;
            if (particle.y < 0) particle.y = height;
            if (particle.y > height) particle.y = 0;

            // Gentle pulse brightness - slow sine wave only, no beat response
            particle.pulse += deltaTime * 0.001;
            const pulseBrightness = Math.sin(particle.pulse) * 0.15 + 0.85;

            // Draw particle - subtle and consistent
            const alpha = particle.brightness * pulseBrightness * reactivity * 0.5;
            const size = particle.size; // Fixed size, no beat reaction

            ctx.beginPath();
            ctx.arc(particle.x, particle.y, size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${Math.min(0.4, alpha)})`;
            ctx.fill();

            // Very subtle glow always present
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, size * 2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${Math.min(0.08, alpha * 0.15)})`;
            ctx.fill();
        });

        ctx.restore();
    }

    renderPulseRings() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const config = this.config;

        const centerX = width / 2;
        const centerY = height / 2;
        const maxRadius = Math.max(width, height) * 0.4;

        // Very subtle expanding ring - single ring only
        const baseAlpha = this.bgState.pulseIntensity * 0.04; // Very subtle

        if (baseAlpha < 0.005) return; // Skip if too faint

        ctx.save();

        // Single expanding ring
        const phase = 1 - this.bgState.pulseIntensity;
        const radius = phase * maxRadius;
        const alpha = baseAlpha * (1 - phase); // Linear falloff

        if (radius > 10) {
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
            ctx.strokeStyle = config.accentColor;
            ctx.lineWidth = 1;
            ctx.globalAlpha = alpha;
            ctx.stroke();
        }

        ctx.restore();
    }

    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    }

    renderFrame(frameData, deltaTime) {
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

        // Smooth all incoming values for fluid motion - very gradual changes
        const smoothFactor = Math.min(1, deltaTime * 0.002); // Even slower smoothing

        // Percussive impact: shapes respond to beats, but not the background
        this.smoothedValues.percussiveImpact = this.lerp(
            this.smoothedValues.percussiveImpact,
            frameData.percussive_impact,
            frameData.is_beat ? 0.15 : smoothFactor * 2 // Gentler beat response
        );

        // Harmonic energy: very slow changes for background stability
        this.smoothedValues.harmonicEnergy = this.lerp(
            this.smoothedValues.harmonicEnergy,
            frameData.harmonic_energy,
            smoothFactor * 0.15 // Ultra slow for background
        );

        // Spectral brightness: slow changes
        this.smoothedValues.spectralBrightness = this.lerp(
            this.smoothedValues.spectralBrightness,
            frameData.spectral_brightness,
            smoothFactor * 0.2
        );

        // Update background state
        this.bgState.gradientAngle += deltaTime * 0.00008 * (1 + this.smoothedValues.harmonicEnergy * 0.5);

        // Pulse rings only - much gentler, doesn't affect background color
        const targetPulse = frameData.is_beat ? 0.4 : 0;
        const pulseLerp = frameData.is_beat ? 0.1 : 0.05; // Very gentle attack and decay
        this.bgState.pulseIntensity = this.lerp(this.bgState.pulseIntensity, targetPulse, pulseLerp);

        this.bgState.noiseOffset += deltaTime * 0.01;

        // Render dynamic background
        if (config.dynamicBg) {
            this.renderDynamicBackground(frameData, deltaTime);
        } else {
            // Simple fade for trail effect
            const fadeAmount = (100 - config.trailAlpha) / 100;
            ctx.fillStyle = config.bgColor;
            ctx.globalAlpha = fadeAmount * 0.3 + 0.02;
            ctx.fillRect(0, 0, width, height);
            ctx.globalAlpha = 1;
        }

        const centerX = width / 2;
        const centerY = height / 2;

        // Calculate visual parameters from smoothed audio values
        const scale = 1 + (this.smoothedValues.percussiveImpact * (config.maxScale - 1));
        const radius = config.baseRadius * scale;

        // Rotation accumulation (time-based, not frame-based)
        const rotationDelta = this.smoothedValues.harmonicEnergy * config.rotationSpeed * (deltaTime / 1000);
        if (this.isPlaying || this.smoothedValues.harmonicEnergy > 0.1) {
            this.accumulatedRotation += rotationDelta;
        } else {
            // Gentle idle rotation when not playing
            this.accumulatedRotation += 0.001 * deltaTime;
        }

        // Polygon sides based on brightness (smoothed)
        const numSides = Math.round(
            config.minSides + this.smoothedValues.spectralBrightness * (config.maxSides - config.minSides)
        );

        // Thickness based on percussive impact (smoothed)
        const thickness = config.baseThickness +
            this.smoothedValues.percussiveImpact * (config.maxThickness - config.baseThickness);

        // Get color
        let hue;
        if (config.chromaColors) {
            hue = this.chromaToHue[frameData.dominant_chroma] || 0;
        } else {
            hue = this.hexToHsl(config.accentColor).h;
        }

        // Render based on selected style
        switch (config.style) {
            case 'glass':
                this.renderGlassStyle(ctx, centerX, centerY, radius, numSides, hue, thickness);
                break;
            case 'flower':
                this.renderFlowerStyle(ctx, centerX, centerY, radius, numSides, hue, thickness);
                break;
            case 'spiral':
                this.renderSpiralStyle(ctx, centerX, centerY, radius, numSides, hue, thickness);
                break;
            case 'geometric':
            default:
                this.renderGeometricStyle(ctx, centerX, centerY, radius, numSides, hue, thickness);
                break;
        }
    }

    /**
     * Geometric style - orbiting polygons with radial symmetry
     */
    renderGeometricStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const orbitDistance = config.orbitRadius * (0.5 + this.smoothedValues.harmonicEnergy * 0.5);

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

    /**
     * Seeded random number generator for consistent patterns
     */
    seededRandom(seed) {
        const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
        return x - Math.floor(x);
    }

    /**
     * Glass style - over-the-top broken glass kaleidoscope with tons of solid geometric shapes
     */
    renderGlassStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const mirrors = config.mirrors;
        const wedgeAngle = (Math.PI * 2) / mirrors;
        const maxRadius = Math.max(this.canvas.width, this.canvas.height) * 0.55;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;

        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(this.accumulatedRotation * 0.15);

        // Draw content in each wedge with mirroring
        for (let i = 0; i < mirrors; i++) {
            ctx.save();
            ctx.rotate(wedgeAngle * i);

            // Create clipping path for wedge
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(maxRadius, 0);
            ctx.arc(0, 0, maxRadius, 0, wedgeAngle / 2);
            ctx.lineTo(0, 0);
            ctx.clip();

            // Draw ALL the glass content
            this.drawGlassContent(ctx, maxRadius, hue, thickness, i);

            ctx.restore();

            // Draw mirrored version
            ctx.save();
            ctx.rotate(wedgeAngle * i);
            ctx.scale(1, -1);

            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(maxRadius, 0);
            ctx.arc(0, 0, maxRadius, 0, wedgeAngle / 2);
            ctx.lineTo(0, 0);
            ctx.clip();

            this.drawGlassContent(ctx, maxRadius, hue, thickness, i + mirrors);

            ctx.restore();
        }

        ctx.restore();

        // Central jewel - larger and more elaborate
        this.drawCentralJewel(ctx, centerX, centerY, radius * 0.4, hue, energy);
    }

    drawGlassContent(ctx, maxRadius, hue, thickness, seedOffset) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;

        // Generate LOTS of shapes with seeded randomness
        const numShapes = 25 + Math.floor(brightness * 20);
        const baseSeed = Math.floor(this.accumulatedRotation * 2) + seedOffset * 100;

        // Layer 1: Large background triangles (solid filled)
        for (let i = 0; i < 8; i++) {
            const seed = baseSeed + i * 7.3;
            const dist = maxRadius * (0.3 + this.seededRandom(seed) * 0.6);
            const angle = this.seededRandom(seed + 1) * 0.4;
            const size = (40 + this.seededRandom(seed + 2) * 80) * (1 + energy * 0.5);
            const shapeHue = (hue + this.seededRandom(seed + 3) * 120 - 60) % 360;

            const x = dist * Math.cos(angle);
            const y = dist * Math.sin(angle) * 0.5;

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(this.seededRandom(seed + 4) * Math.PI + this.accumulatedRotation * 0.3);

            // Solid filled triangle
            ctx.beginPath();
            ctx.moveTo(0, -size);
            ctx.lineTo(size * 0.866, size * 0.5);
            ctx.lineTo(-size * 0.866, size * 0.5);
            ctx.closePath();

            const alpha = 0.15 + harmonic * 0.2;
            ctx.fillStyle = `hsla(${shapeHue}, ${config.saturation}%, 50%, ${alpha})`;
            ctx.fill();
            ctx.strokeStyle = `hsla(${shapeHue}, ${config.saturation}%, 70%, ${alpha + 0.2})`;
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.restore();
        }

        // Layer 2: Medium diamonds and rectangles
        for (let i = 0; i < 12; i++) {
            const seed = baseSeed + 100 + i * 11.7;
            const dist = maxRadius * (0.15 + this.seededRandom(seed) * 0.7);
            const angle = this.seededRandom(seed + 1) * 0.45;
            const size = (20 + this.seededRandom(seed + 2) * 50) * (1 + energy * 0.6);
            const shapeHue = (hue + this.seededRandom(seed + 3) * 80 - 40) % 360;
            const shapeType = Math.floor(this.seededRandom(seed + 5) * 3);

            const x = dist * Math.cos(angle);
            const y = dist * Math.sin(angle) * 0.45;

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(this.seededRandom(seed + 4) * Math.PI * 2 + this.accumulatedRotation * (0.2 + this.seededRandom(seed + 6) * 0.3));

            ctx.beginPath();
            if (shapeType === 0) {
                // Diamond
                ctx.moveTo(0, -size);
                ctx.lineTo(size * 0.6, 0);
                ctx.lineTo(0, size);
                ctx.lineTo(-size * 0.6, 0);
            } else if (shapeType === 1) {
                // Rectangle
                ctx.rect(-size * 0.4, -size * 0.7, size * 0.8, size * 1.4);
            } else {
                // Hexagon
                for (let j = 0; j < 6; j++) {
                    const a = (Math.PI * 2 * j) / 6;
                    const px = Math.cos(a) * size * 0.6;
                    const py = Math.sin(a) * size * 0.6;
                    if (j === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
            }
            ctx.closePath();

            const alpha = 0.25 + energy * 0.25 + harmonic * 0.15;
            ctx.fillStyle = `hsla(${shapeHue}, ${config.saturation * 0.9}%, 55%, ${alpha})`;
            ctx.fill();
            ctx.strokeStyle = `hsla(${shapeHue}, ${config.saturation}%, 75%, ${alpha + 0.15})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            ctx.restore();
        }

        // Layer 3: Small accent triangles (many!)
        for (let i = 0; i < numShapes; i++) {
            const seed = baseSeed + 200 + i * 5.1;
            const dist = maxRadius * (0.1 + this.seededRandom(seed) * 0.85);
            const angle = this.seededRandom(seed + 1) * 0.5;
            const size = (8 + this.seededRandom(seed + 2) * 25) * (1 + energy * 0.8);
            const shapeHue = (hue + this.seededRandom(seed + 3) * 60) % 360;

            const x = dist * Math.cos(angle);
            const y = dist * Math.sin(angle) * 0.4;

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(this.seededRandom(seed + 4) * Math.PI * 2 + this.accumulatedRotation * this.seededRandom(seed + 5));

            // Small solid triangle
            ctx.beginPath();
            ctx.moveTo(0, -size);
            ctx.lineTo(size * 0.7, size * 0.6);
            ctx.lineTo(-size * 0.7, size * 0.6);
            ctx.closePath();

            const alpha = 0.35 + energy * 0.35;
            ctx.fillStyle = `hsla(${shapeHue}, ${config.saturation}%, 60%, ${alpha})`;
            ctx.fill();

            ctx.restore();
        }

        // Layer 4: Bright highlight shards
        for (let i = 0; i < 8; i++) {
            const seed = baseSeed + 300 + i * 13.3;
            const dist = maxRadius * (0.2 + this.seededRandom(seed) * 0.6);
            const angle = this.seededRandom(seed + 1) * 0.4;
            const size = (15 + this.seededRandom(seed + 2) * 35) * (1 + energy * 0.5);
            const shapeHue = (hue + 30 + this.seededRandom(seed + 3) * 40) % 360;

            const x = dist * Math.cos(angle);
            const y = dist * Math.sin(angle) * 0.35;

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(this.seededRandom(seed + 4) * Math.PI + this.accumulatedRotation * 0.5);

            // Elongated crystal shard
            ctx.beginPath();
            ctx.moveTo(0, -size * 1.5);
            ctx.lineTo(size * 0.3, 0);
            ctx.lineTo(0, size);
            ctx.lineTo(-size * 0.3, 0);
            ctx.closePath();

            const alpha = 0.5 + energy * 0.3;
            ctx.fillStyle = `hsla(${shapeHue}, ${config.saturation * 0.8}%, 75%, ${alpha})`;
            ctx.fill();
            ctx.strokeStyle = `hsla(${shapeHue}, ${config.saturation}%, 90%, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.restore();
        }

        // Radial fracture lines
        const numLines = 5 + Math.floor(brightness * 4);
        for (let i = 0; i < numLines; i++) {
            const seed = baseSeed + 400 + i;
            const lineAngle = this.seededRandom(seed) * 0.5;
            const lineHue = (hue + i * 25) % 360;

            ctx.beginPath();
            ctx.moveTo(0, 0);
            const endX = maxRadius * Math.cos(lineAngle);
            const endY = maxRadius * Math.sin(lineAngle) * 0.45;
            ctx.lineTo(endX, endY);
            ctx.strokeStyle = `hsla(${lineHue}, ${config.saturation * 0.6}%, 65%, 0.25)`;
            ctx.lineWidth = 1 + energy;
            ctx.stroke();
        }
    }

    drawCentralJewel(ctx, centerX, centerY, radius, hue, energy) {
        const config = this.config;

        // Outer glow
        const glowGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius * 1.5);
        glowGradient.addColorStop(0, `hsla(${hue}, ${config.saturation}%, 80%, 0.4)`);
        glowGradient.addColorStop(0.5, `hsla(${hue}, ${config.saturation}%, 60%, 0.2)`);
        glowGradient.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 40%, 0)`);
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * 1.5, 0, Math.PI * 2);
        ctx.fill();

        // Multi-faceted jewel
        const facets = 8;
        for (let i = 0; i < facets; i++) {
            const angle = (Math.PI * 2 * i) / facets + this.accumulatedRotation * 0.3;
            const nextAngle = (Math.PI * 2 * (i + 1)) / facets + this.accumulatedRotation * 0.3;

            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(
                centerX + Math.cos(angle) * radius,
                centerY + Math.sin(angle) * radius
            );
            ctx.lineTo(
                centerX + Math.cos(nextAngle) * radius,
                centerY + Math.sin(nextAngle) * radius
            );
            ctx.closePath();

            const facetHue = (hue + i * 15) % 360;
            const lightness = 50 + (i % 2) * 20 + energy * 15;
            ctx.fillStyle = `hsla(${facetHue}, ${config.saturation}%, ${lightness}%, 0.7)`;
            ctx.fill();
            ctx.strokeStyle = `hsla(${facetHue}, ${config.saturation}%, 85%, 0.8)`;
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // Inner core
        const coreGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius * 0.4);
        coreGradient.addColorStop(0, `hsla(${hue}, ${config.saturation}%, 95%, 1)`);
        coreGradient.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 70%, 0.5)`);
        ctx.fillStyle = coreGradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * 0.4, 0, Math.PI * 2);
        ctx.fill();
    }

    /**
     * Flower style - petal-like shapes radiating from center
     */
    renderFlowerStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const mirrors = config.mirrors;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;

        ctx.save();
        ctx.translate(centerX, centerY);

        // Multiple layers of petals
        const layers = 3;
        for (let layer = 0; layer < layers; layer++) {
            const layerRadius = radius * (0.5 + layer * 0.4) * (1 + energy * 0.3);
            const layerRotation = this.accumulatedRotation * (1 - layer * 0.3) * (layer % 2 === 0 ? 1 : -1);
            const petalCount = mirrors + layer * 2;
            const layerHue = (hue + layer * 40) % 360;

            ctx.save();
            ctx.rotate(layerRotation);

            for (let i = 0; i < petalCount; i++) {
                const petalAngle = (Math.PI * 2 * i) / petalCount;
                const petalLength = layerRadius * (0.8 + harmonic * 0.4);
                const petalWidth = layerRadius * 0.3 * (1 + energy * 0.5);

                ctx.save();
                ctx.rotate(petalAngle);

                // Draw petal shape using bezier curves
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.bezierCurveTo(
                    petalWidth, petalLength * 0.3,
                    petalWidth * 0.5, petalLength * 0.8,
                    0, petalLength
                );
                ctx.bezierCurveTo(
                    -petalWidth * 0.5, petalLength * 0.8,
                    -petalWidth, petalLength * 0.3,
                    0, 0
                );

                const alpha = 0.4 + brightness * 0.3 - layer * 0.1;
                ctx.strokeStyle = `hsla(${layerHue}, ${config.saturation}%, ${60 + layer * 10}%, ${alpha})`;
                ctx.lineWidth = thickness * (1 - layer * 0.2);
                ctx.stroke();

                // Inner vein
                ctx.beginPath();
                ctx.moveTo(0, petalLength * 0.1);
                ctx.lineTo(0, petalLength * 0.8);
                ctx.strokeStyle = `hsla(${layerHue}, ${config.saturation}%, 70%, ${alpha * 0.5})`;
                ctx.lineWidth = thickness * 0.3;
                ctx.stroke();

                ctx.restore();
            }

            ctx.restore();
        }

        // Center stamen
        const stamenRadius = radius * 0.15 * (1 + energy * 0.5);
        const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, stamenRadius);
        gradient.addColorStop(0, `hsla(${(hue + 60) % 360}, ${config.saturation}%, 80%, 0.9)`);
        gradient.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 50%, 0)`);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(0, 0, stamenRadius, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    /**
     * Spiral style - fractal spirals with branching sub-spirals
     */
    renderSpiralStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;

        ctx.save();
        ctx.translate(centerX, centerY);

        const arms = config.mirrors;
        const maxRadius = radius * 3;

        // Draw multiple layers of fractal spirals
        this.drawFractalSpirals(ctx, arms, maxRadius, hue, thickness, 0, 1.0);

        // Second layer - counter-rotating
        ctx.save();
        ctx.rotate(Math.PI / arms);
        this.drawFractalSpirals(ctx, arms, maxRadius * 0.7, (hue + 60) % 360, thickness * 0.7, 1, 0.6);
        ctx.restore();

        // Third layer - smaller, faster
        ctx.save();
        ctx.rotate(this.accumulatedRotation * 0.5);
        this.drawFractalSpirals(ctx, arms * 2, maxRadius * 0.4, (hue + 120) % 360, thickness * 0.5, 2, 0.4);
        ctx.restore();

        // Central mandala
        this.drawSpiralCore(ctx, radius * 0.35, hue, energy, harmonic);

        ctx.restore();
    }

    drawFractalSpirals(ctx, arms, maxRadius, hue, thickness, layer, opacity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;

        const pointsPerArm = 35 + Math.floor(brightness * 25);
        const spiralTightness = 2.5 + harmonic * 1.5;
        const rotationOffset = this.accumulatedRotation * (layer % 2 === 0 ? 1 : -0.7);

        for (let arm = 0; arm < arms; arm++) {
            const armAngle = (Math.PI * 2 * arm) / arms + rotationOffset;
            const armHue = (hue + arm * (180 / arms)) % 360;

            // Main spiral arm
            ctx.beginPath();
            let prevX = 0, prevY = 0;

            for (let i = 0; i < pointsPerArm; i++) {
                const t = i / pointsPerArm;
                const spiralAngle = armAngle + t * Math.PI * spiralTightness;
                const spiralRadius = t * maxRadius * (0.3 + harmonic * 0.7);

                const x = spiralRadius * Math.cos(spiralAngle);
                const y = spiralRadius * Math.sin(spiralAngle);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }

                prevX = x;
                prevY = y;

                // Branch points - spawn sub-spirals
                if (i > 5 && i % 6 === 0 && layer < 2) {
                    this.drawSpiralBranch(ctx, x, y, spiralAngle, t, armHue, thickness, maxRadius, layer);
                }
            }

            const alpha = (0.6 + energy * 0.3) * opacity;
            ctx.strokeStyle = `hsla(${armHue}, ${config.saturation}%, 65%, ${alpha})`;
            ctx.lineWidth = thickness * (1 + energy * 0.5) * (1 - layer * 0.2);
            ctx.lineCap = 'round';
            ctx.stroke();

            // Nodes along main spiral
            for (let i = 0; i < pointsPerArm; i += 4) {
                const t = i / pointsPerArm;
                const spiralAngle = armAngle + t * Math.PI * spiralTightness;
                const spiralRadius = t * maxRadius * (0.3 + harmonic * 0.7);

                const x = spiralRadius * Math.cos(spiralAngle);
                const y = spiralRadius * Math.sin(spiralAngle);

                const nodeSize = (3 + energy * 6) * (1 - t * 0.3) * (1 - layer * 0.3);
                const nodeHue = (armHue + t * 40) % 360;

                // Outer glow
                ctx.beginPath();
                ctx.arc(x, y, nodeSize * 2, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${nodeHue}, ${config.saturation}%, 70%, ${0.15 * opacity})`;
                ctx.fill();

                // Inner node
                ctx.beginPath();
                ctx.arc(x, y, nodeSize, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${nodeHue}, ${config.saturation}%, 80%, ${(0.7 + energy * 0.3) * opacity})`;
                ctx.fill();
            }
        }
    }

    drawSpiralBranch(ctx, startX, startY, parentAngle, parentT, hue, thickness, maxRadius, layer) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;

        const branchLength = maxRadius * 0.25 * (1 - parentT * 0.5);
        const branchPoints = 12;
        const branchAngleOffset = (this.seededRandom(parentAngle * 100) - 0.5) * Math.PI * 0.8;

        ctx.save();
        ctx.translate(startX, startY);
        ctx.rotate(parentAngle + branchAngleOffset + Math.PI / 2);

        // Draw branch spiral
        ctx.beginPath();
        for (let i = 0; i < branchPoints; i++) {
            const t = i / branchPoints;
            const branchAngle = t * Math.PI * 1.5;
            const branchRadius = t * branchLength;

            const x = branchRadius * Math.cos(branchAngle);
            const y = branchRadius * Math.sin(branchAngle);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        const branchHue = (hue + 30) % 360;
        const alpha = 0.4 + energy * 0.2;
        ctx.strokeStyle = `hsla(${branchHue}, ${config.saturation * 0.8}%, 60%, ${alpha})`;
        ctx.lineWidth = thickness * 0.4;
        ctx.stroke();

        // Tiny nodes on branch
        for (let i = 3; i < branchPoints; i += 3) {
            const t = i / branchPoints;
            const branchAngle = t * Math.PI * 1.5;
            const branchRadius = t * branchLength;

            const x = branchRadius * Math.cos(branchAngle);
            const y = branchRadius * Math.sin(branchAngle);

            ctx.beginPath();
            ctx.arc(x, y, 2 + energy * 2, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${branchHue}, ${config.saturation}%, 75%, ${0.5 + energy * 0.3})`;
            ctx.fill();
        }

        // Sub-branches (recursive feel without actual recursion)
        if (layer === 0) {
            const subBranchLength = branchLength * 0.4;
            for (let i = 0; i < 2; i++) {
                const t = 0.4 + i * 0.3;
                const branchAngle = t * Math.PI * 1.5;
                const branchRadius = t * branchLength;
                const bx = branchRadius * Math.cos(branchAngle);
                const by = branchRadius * Math.sin(branchAngle);

                ctx.beginPath();
                ctx.moveTo(bx, by);
                const subAngle = branchAngle + (i === 0 ? 0.8 : -0.8);
                ctx.lineTo(
                    bx + Math.cos(subAngle) * subBranchLength,
                    by + Math.sin(subAngle) * subBranchLength
                );
                ctx.strokeStyle = `hsla(${(branchHue + 20) % 360}, ${config.saturation * 0.7}%, 55%, ${alpha * 0.6})`;
                ctx.lineWidth = thickness * 0.2;
                ctx.stroke();
            }
        }

        ctx.restore();
    }

    drawSpiralCore(ctx, radius, hue, energy, harmonic) {
        const config = this.config;

        // Rotating inner spirals
        const innerArms = 6;
        for (let i = 0; i < innerArms; i++) {
            const armAngle = (Math.PI * 2 * i) / innerArms + this.accumulatedRotation * 2;

            ctx.beginPath();
            for (let j = 0; j < 15; j++) {
                const t = j / 15;
                const spiralAngle = armAngle + t * Math.PI * 2;
                const spiralRadius = t * radius;

                const x = spiralRadius * Math.cos(spiralAngle);
                const y = spiralRadius * Math.sin(spiralAngle);

                if (j === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            const armHue = (hue + i * 30) % 360;
            ctx.strokeStyle = `hsla(${armHue}, ${config.saturation}%, 70%, ${0.6 + energy * 0.3})`;
            ctx.lineWidth = 2 + energy * 2;
            ctx.stroke();
        }

        // Concentric rings
        for (let ring = 0; ring < 4; ring++) {
            const ringRadius = radius * (0.2 + ring * 0.25) * (1 + energy * 0.2);
            ctx.beginPath();
            ctx.arc(0, 0, ringRadius, 0, Math.PI * 2);
            const ringHue = (hue + ring * 20) % 360;
            ctx.strokeStyle = `hsla(${ringHue}, ${config.saturation}%, ${65 + ring * 8}%, ${0.5 - ring * 0.1 + energy * 0.2})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        // Central glow
        const glowGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, radius * 0.5);
        glowGradient.addColorStop(0, `hsla(${hue}, ${config.saturation}%, 90%, ${0.8 + energy * 0.2})`);
        glowGradient.addColorStop(0.5, `hsla(${hue}, ${config.saturation}%, 70%, 0.3)`);
        glowGradient.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 50%, 0)`);
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(0, 0, radius * 0.5, 0, Math.PI * 2);
        ctx.fill();
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
        this.lastFrameTime = performance.now();
        this.renderFrame(null, 16.67); // ~60fps delta
    }

    toggleFullscreen() {
        const container = document.querySelector('.canvas-container');

        if (!document.fullscreenElement) {
            container.requestFullscreen().catch(err => {
                console.log('Fullscreen error:', err);
            });
        } else {
            document.exitFullscreen();
        }
    }

    async exportVideo() {
        if (!this.audioBuffer) {
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
