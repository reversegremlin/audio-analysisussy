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
            shapeSeed: Math.floor(Math.random() * 10000), // Random seed for shape generation
            glassSlices: 30, // Number of shape slices in Glass style (10-60)
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
        const glassSlicesControl = document.getElementById('glassSlicesControl');
        document.querySelectorAll('.style-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.config.style = btn.dataset.style;
                // Show/hide glass-specific controls
                if (glassSlicesControl) {
                    if (btn.dataset.style === 'glass') {
                        glassSlicesControl.classList.add('visible');
                    } else {
                        glassSlicesControl.classList.remove('visible');
                    }
                }
            });
        });

        // Glass slices slider
        const glassSlicesSlider = document.getElementById('glassSlicesSlider');
        const glassSlicesValue = document.getElementById('glassSlicesValue');
        if (glassSlicesSlider) {
            glassSlicesSlider.addEventListener('input', (e) => {
                this.config.glassSlices = parseInt(e.target.value, 10);
                if (glassSlicesValue) {
                    glassSlicesValue.textContent = e.target.value;
                }
            });
        }

        // Randomize button
        const randomizeBtn = document.getElementById('randomizeBtn');
        if (randomizeBtn) {
            randomizeBtn.addEventListener('click', () => {
                this.config.shapeSeed = Math.floor(Math.random() * 10000);
                // Visual feedback
                randomizeBtn.classList.add('clicked');
                setTimeout(() => randomizeBtn.classList.remove('clicked'), 200);
            });
        }

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
     * Geometric style - orbiting polygons with radial symmetry (uses shapeSeed for variation)
     */
    renderGeometricStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const orbitDistance = config.orbitRadius * (0.5 + harmonic * 0.5);

        // Seed-based variation parameters
        const rotationDir = this.seededRandom(seed) > 0.5 ? 1 : -1;
        const innerRotationSpeed = 1 + this.seededRandom(seed + 1) * 1.5;
        const hueShift = this.seededRandom(seed + 2) * 60;
        const sizeVariation = 0.7 + this.seededRandom(seed + 3) * 0.6;

        // Draw kaleidoscope pattern
        for (let i = 0; i < config.mirrors; i++) {
            const iSeed = seed + i * 17;
            const mirrorAngle = (Math.PI * 2 * i / config.mirrors) + this.accumulatedRotation * 0.3 * rotationDir;

            // Seed-based per-mirror variation
            const orbitVar = 0.9 + this.seededRandom(iSeed) * 0.2;
            const sizeVar = 0.8 + this.seededRandom(iSeed + 1) * 0.4;
            const sidesVar = Math.floor(this.seededRandom(iSeed + 2) * 3);

            const orbitX = centerX + orbitDistance * orbitVar * Math.cos(mirrorAngle);
            const orbitY = centerY + orbitDistance * orbitVar * Math.sin(mirrorAngle);

            // Outer polygon with seed variation
            const outerSides = numSides + sidesVar;
            const outerHue = (hue + i * (hueShift / config.mirrors)) % 360;
            this.drawPolygon(
                ctx,
                orbitX, orbitY,
                radius * 0.8 * sizeVar * sizeVariation,
                outerSides,
                this.accumulatedRotation * rotationDir + mirrorAngle,
                `hsl(${outerHue}, ${config.saturation}%, ${65 + energy * 15}%)`,
                thickness * (0.8 + energy * 0.4)
            );

            // Inner polygon (counter-rotating) with variation
            const innerHue = (hue + 180 + hueShift) % 360;
            const innerSides = Math.max(3, outerSides - 1 - Math.floor(this.seededRandom(iSeed + 3) * 2));
            this.drawPolygon(
                ctx,
                orbitX, orbitY,
                radius * 0.35 * sizeVar,
                innerSides,
                -this.accumulatedRotation * innerRotationSpeed * rotationDir + mirrorAngle,
                `hsl(${innerHue}, ${config.saturation * 0.8}%, ${55 + energy * 20}%)`,
                Math.max(1, thickness / 2)
            );

            // Extra tiny polygon on some mirrors (seed-based)
            if (this.seededRandom(iSeed + 4) > 0.5) {
                const tinyHue = (outerHue + 90) % 360;
                this.drawPolygon(
                    ctx,
                    orbitX, orbitY,
                    radius * 0.15 * (1 + energy * 0.5),
                    3 + Math.floor(this.seededRandom(iSeed + 5) * 3),
                    this.accumulatedRotation * 2 * rotationDir,
                    `hsl(${tinyHue}, ${config.saturation}%, ${70 + brightness * 20}%)`,
                    1 + energy
                );
            }
        }

        // Central polygon with seed-based sides
        const centralSides = numSides + Math.floor(this.seededRandom(seed + 10) * 3);
        this.drawPolygon(
            ctx,
            centerX, centerY,
            radius * 0.6 * (1 + energy * 0.2),
            centralSides,
            this.accumulatedRotation * 0.5 * rotationDir,
            `hsl(${(hue + hueShift) % 360}, ${config.saturation}%, ${75 + energy * 15}%)`,
            thickness + 2
        );

        // Second central layer (seed-based)
        if (this.seededRandom(seed + 11) > 0.3) {
            this.drawPolygon(
                ctx,
                centerX, centerY,
                radius * 0.35,
                Math.max(3, centralSides - 2),
                -this.accumulatedRotation * 0.8 * rotationDir,
                `hsl(${(hue + 120) % 360}, ${config.saturation * 0.9}%, ${65 + harmonic * 20}%)`,
                thickness
            );
        }
    }

    /**
     * Seeded random number generator for consistent patterns
     */
    seededRandom(seed) {
        const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
        return x - Math.floor(x);
    }

    /**
     * Glass style - broken glass kaleidoscope with reactive geometric shapes
     */
    renderGlassStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const mirrors = config.mirrors * 2; // Double for better symmetry
        const maxRadius = Math.min(this.canvas.width, this.canvas.height) * 0.48;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const seed = config.shapeSeed;

        ctx.save();
        ctx.translate(centerX, centerY);

        // Global rotation based on harmonic energy
        const globalRotation = this.accumulatedRotation * 0.3;
        ctx.rotate(globalRotation);

        // Draw mirrored wedges
        for (let m = 0; m < mirrors; m++) {
            ctx.save();
            ctx.rotate((Math.PI * 2 * m) / mirrors);

            // Mirror every other wedge
            if (m % 2 === 1) {
                ctx.scale(-1, 1);
            }

            // Draw shapes in this wedge
            this.drawGlassWedge(ctx, maxRadius, hue, thickness, seed, m);

            ctx.restore();
        }

        ctx.restore();

        // Pulsing central jewel
        const jewelSize = radius * (0.25 + energy * 0.2);
        this.drawCentralJewel(ctx, centerX, centerY, jewelSize, hue, energy);
    }

    drawGlassWedge(ctx, maxRadius, hue, thickness, baseSeed, wedgeIndex) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;

        // Wedge angle for clipping
        const wedgeAngle = Math.PI / config.mirrors;

        // Clip to wedge
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, maxRadius, 0, wedgeAngle);
        ctx.closePath();
        ctx.clip();

        const numShapes = config.glassSlices + Math.floor(brightness * 25);

        // Layer 1: Large pulsing background shapes
        for (let i = 0; i < 6; i++) {
            const seed = baseSeed + i * 17.3;
            const baseSize = 50 + this.seededRandom(seed) * 100;
            const size = baseSize * (0.8 + energy * 0.6 + harmonic * 0.3); // REACTIVE SIZE
            const dist = maxRadius * (0.25 + this.seededRandom(seed + 1) * 0.55);
            const angle = this.seededRandom(seed + 2) * wedgeAngle * 0.9;
            const shapeHue = (hue + this.seededRandom(seed + 3) * 80 - 40 + harmonic * 30) % 360;
            const rotation = this.seededRandom(seed + 4) * Math.PI * 2 + this.accumulatedRotation * (0.5 + this.seededRandom(seed + 5) * 0.5);

            const x = Math.cos(angle) * dist;
            const y = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(rotation);

            // Random polygon (3-6 sides)
            const sides = 3 + Math.floor(this.seededRandom(seed + 6) * 4);
            ctx.beginPath();
            for (let j = 0; j < sides; j++) {
                const a = (Math.PI * 2 * j) / sides - Math.PI / 2;
                const px = Math.cos(a) * size * 0.5;
                const py = Math.sin(a) * size * 0.5;
                if (j === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.closePath();

            const alpha = 0.2 + energy * 0.3 + harmonic * 0.15;
            ctx.fillStyle = `hsla(${shapeHue}, ${config.saturation}%, ${45 + brightness * 20}%, ${alpha})`;
            ctx.fill();
            ctx.strokeStyle = `hsla(${shapeHue}, ${config.saturation}%, 70%, ${alpha + 0.2})`;
            ctx.lineWidth = 1 + energy * 2;
            ctx.stroke();

            ctx.restore();
        }

        // Layer 2: Medium reactive diamonds
        for (let i = 0; i < 10; i++) {
            const seed = baseSeed + 100 + i * 23.7;
            const baseSize = 25 + this.seededRandom(seed) * 50;
            const size = baseSize * (0.7 + energy * 0.8); // REACTIVE
            const dist = maxRadius * (0.15 + this.seededRandom(seed + 1) * 0.7);
            const angle = this.seededRandom(seed + 2) * wedgeAngle * 0.95;
            const shapeHue = (hue + this.seededRandom(seed + 3) * 60 - 30 + brightness * 40) % 360;
            const rotation = this.seededRandom(seed + 4) * Math.PI + this.accumulatedRotation * (0.3 + energy * 0.4);

            const x = Math.cos(angle) * dist;
            const y = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(rotation);

            // Diamond shape
            ctx.beginPath();
            ctx.moveTo(0, -size);
            ctx.lineTo(size * 0.5, 0);
            ctx.lineTo(0, size * 0.7);
            ctx.lineTo(-size * 0.5, 0);
            ctx.closePath();

            const alpha = 0.3 + energy * 0.4;
            ctx.fillStyle = `hsla(${shapeHue}, ${config.saturation * 0.9}%, ${50 + energy * 20}%, ${alpha})`;
            ctx.fill();
            ctx.strokeStyle = `hsla(${shapeHue}, ${config.saturation}%, 80%, ${alpha + 0.2})`;
            ctx.lineWidth = 1.5 + energy;
            ctx.stroke();

            ctx.restore();
        }

        // Layer 3: Small sparkle triangles (LOTS!)
        for (let i = 0; i < numShapes; i++) {
            const seed = baseSeed + 200 + i * 7.1;
            const baseSize = 8 + this.seededRandom(seed) * 20;
            const size = baseSize * (0.5 + energy * 1.2 + brightness * 0.3); // VERY REACTIVE
            const dist = maxRadius * (0.08 + this.seededRandom(seed + 1) * 0.88);
            const angle = this.seededRandom(seed + 2) * wedgeAngle;
            const shapeHue = (hue + this.seededRandom(seed + 3) * 50 + harmonic * 50) % 360;
            const rotation = this.seededRandom(seed + 4) * Math.PI * 2 + this.accumulatedRotation * this.seededRandom(seed + 5) * 2;

            const x = Math.cos(angle) * dist;
            const y = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(rotation);

            ctx.beginPath();
            ctx.moveTo(0, -size);
            ctx.lineTo(size * 0.6, size * 0.5);
            ctx.lineTo(-size * 0.6, size * 0.5);
            ctx.closePath();

            const alpha = 0.4 + energy * 0.5;
            ctx.fillStyle = `hsla(${shapeHue}, ${config.saturation}%, ${55 + brightness * 25}%, ${alpha})`;
            ctx.fill();

            ctx.restore();
        }

        // Layer 4: Bright crystal highlights (pulse with beats)
        for (let i = 0; i < 5; i++) {
            const seed = baseSeed + 300 + i * 31.3;
            const baseSize = 15 + this.seededRandom(seed) * 30;
            const size = baseSize * (0.6 + energy * 1.0); // BIG PULSE on beats
            const dist = maxRadius * (0.2 + this.seededRandom(seed + 1) * 0.5);
            const angle = this.seededRandom(seed + 2) * wedgeAngle * 0.8;
            const shapeHue = (hue + 20 + this.seededRandom(seed + 3) * 30) % 360;

            const x = Math.cos(angle) * dist;
            const y = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(this.seededRandom(seed + 4) * Math.PI + this.accumulatedRotation * 0.8);

            // Crystal shard
            ctx.beginPath();
            ctx.moveTo(0, -size * 1.5);
            ctx.lineTo(size * 0.35, -size * 0.3);
            ctx.lineTo(size * 0.2, size);
            ctx.lineTo(-size * 0.2, size);
            ctx.lineTo(-size * 0.35, -size * 0.3);
            ctx.closePath();

            const alpha = 0.5 + energy * 0.4;
            ctx.fillStyle = `hsla(${shapeHue}, ${config.saturation * 0.8}%, ${70 + energy * 20}%, ${alpha})`;
            ctx.fill();
            ctx.strokeStyle = `hsla(${shapeHue}, ${config.saturation}%, 95%, ${alpha})`;
            ctx.lineWidth = 1 + energy;
            ctx.stroke();

            ctx.restore();
        }

        // Fracture lines - pulse thickness with energy
        const numLines = 4 + Math.floor(brightness * 3);
        for (let i = 0; i < numLines; i++) {
            const seed = baseSeed + 400 + i * 11;
            const lineAngle = this.seededRandom(seed) * wedgeAngle * 0.9;
            const lineHue = (hue + i * 20) % 360;

            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(Math.cos(lineAngle) * maxRadius, Math.sin(lineAngle) * maxRadius);
            ctx.strokeStyle = `hsla(${lineHue}, ${config.saturation * 0.5}%, 60%, ${0.15 + energy * 0.2})`;
            ctx.lineWidth = 0.5 + energy * 2;
            ctx.stroke();
        }

        ctx.restore();
    }

    drawCentralJewel(ctx, centerX, centerY, radius, hue, energy) {
        const config = this.config;

        // Pulsing outer glow
        const glowSize = radius * (1.5 + energy * 0.5);
        const glowGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, glowSize);
        glowGradient.addColorStop(0, `hsla(${hue}, ${config.saturation}%, ${80 + energy * 15}%, ${0.5 + energy * 0.3})`);
        glowGradient.addColorStop(0.5, `hsla(${hue}, ${config.saturation}%, 60%, ${0.2 + energy * 0.2})`);
        glowGradient.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 40%, 0)`);
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, glowSize, 0, Math.PI * 2);
        ctx.fill();

        // Smooth jewel gradient (no visible facets)
        const jewelGrad = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
        jewelGrad.addColorStop(0, `hsla(${hue}, ${config.saturation}%, ${85 + energy * 15}%, ${0.9 + energy * 0.1})`);
        jewelGrad.addColorStop(0.4, `hsla(${(hue + 20) % 360}, ${config.saturation}%, ${65 + energy * 20}%, ${0.7 + energy * 0.2})`);
        jewelGrad.addColorStop(0.7, `hsla(${(hue + 40) % 360}, ${config.saturation}%, ${50 + energy * 15}%, ${0.5 + energy * 0.2})`);
        jewelGrad.addColorStop(1, `hsla(${(hue + 60) % 360}, ${config.saturation}%, 40%, ${0.2 + energy * 0.1})`);
        ctx.fillStyle = jewelGrad;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.fill();

        // Bright pulsing core
        const coreRadius = radius * (0.35 + energy * 0.15);
        const coreGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, coreRadius);
        coreGradient.addColorStop(0, `hsla(${hue}, ${config.saturation}%, ${95 + energy * 5}%, 1)`);
        coreGradient.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 75%, ${0.4 + energy * 0.4})`);
        ctx.fillStyle = coreGradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, coreRadius, 0, Math.PI * 2);
        ctx.fill();
    }

    /**
     * Flower style - petal-like shapes radiating from center (uses shapeSeed for variation)
     */
    renderFlowerStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const mirrors = config.mirrors;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;

        // Seed-based variation parameters
        const rotationDir = this.seededRandom(seed) > 0.5 ? 1 : -1;
        const petalShape = this.seededRandom(seed + 1); // 0-1 affects curve shape
        const hueSpread = 20 + this.seededRandom(seed + 2) * 60;
        const layerCount = 3 + Math.floor(this.seededRandom(seed + 3) * 2);

        ctx.save();
        ctx.translate(centerX, centerY);

        // Multiple layers of petals with seed variation
        for (let layer = 0; layer < layerCount; layer++) {
            const layerSeed = seed + layer * 31;
            const layerRadius = radius * (0.4 + layer * 0.35) * (1 + energy * 0.3);
            const rotSpeed = (1 - layer * 0.25) * (this.seededRandom(layerSeed) > 0.5 ? 1 : -1);
            const layerRotation = this.accumulatedRotation * rotSpeed * rotationDir;
            const basePetalCount = mirrors + layer * 2;
            const petalCount = basePetalCount + Math.floor(this.seededRandom(layerSeed + 1) * 3);
            const layerHue = (hue + layer * hueSpread) % 360;

            ctx.save();
            ctx.rotate(layerRotation);

            for (let i = 0; i < petalCount; i++) {
                const petalSeed = layerSeed + i * 13;
                const petalAngle = (Math.PI * 2 * i) / petalCount;

                // Seed-based petal size variation
                const sizeVar = 0.8 + this.seededRandom(petalSeed) * 0.4;
                const petalLength = layerRadius * (0.7 + harmonic * 0.5) * sizeVar;
                const petalWidth = layerRadius * (0.25 + petalShape * 0.15) * (1 + energy * 0.5);

                // Seed-based curve variation
                const curveVar1 = 0.2 + this.seededRandom(petalSeed + 1) * 0.2;
                const curveVar2 = 0.7 + this.seededRandom(petalSeed + 2) * 0.2;

                ctx.save();
                ctx.rotate(petalAngle);

                // Draw petal shape using bezier curves with variation
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.bezierCurveTo(
                    petalWidth * (1 + this.seededRandom(petalSeed + 3) * 0.3), petalLength * curveVar1,
                    petalWidth * (0.4 + this.seededRandom(petalSeed + 4) * 0.3), petalLength * curveVar2,
                    0, petalLength
                );
                ctx.bezierCurveTo(
                    -petalWidth * (0.4 + this.seededRandom(petalSeed + 5) * 0.3), petalLength * curveVar2,
                    -petalWidth * (1 + this.seededRandom(petalSeed + 6) * 0.3), petalLength * curveVar1,
                    0, 0
                );

                const alpha = 0.35 + brightness * 0.35 + energy * 0.2 - layer * 0.08;
                const petalHue = (layerHue + this.seededRandom(petalSeed + 7) * 20) % 360;
                ctx.strokeStyle = `hsla(${petalHue}, ${config.saturation}%, ${55 + layer * 8 + energy * 15}%, ${alpha})`;
                ctx.lineWidth = thickness * (1 - layer * 0.15) * (0.8 + energy * 0.4);
                ctx.stroke();

                // Optional fill for some petals (seed-based)
                if (this.seededRandom(petalSeed + 8) > 0.6) {
                    ctx.fillStyle = `hsla(${petalHue}, ${config.saturation * 0.8}%, ${60 + energy * 20}%, ${alpha * 0.3})`;
                    ctx.fill();
                }

                // Inner vein with variation
                if (this.seededRandom(petalSeed + 9) > 0.3) {
                    ctx.beginPath();
                    ctx.moveTo(0, petalLength * 0.15);
                    ctx.quadraticCurveTo(
                        this.seededRandom(petalSeed + 10) * 4 - 2, petalLength * 0.5,
                        0, petalLength * 0.85
                    );
                    ctx.strokeStyle = `hsla(${petalHue}, ${config.saturation}%, 75%, ${alpha * 0.4})`;
                    ctx.lineWidth = thickness * 0.25;
                    ctx.stroke();
                }

                ctx.restore();
            }

            ctx.restore();
        }

        // Center stamen with seed variation
        const stamenRadius = radius * (0.12 + this.seededRandom(seed + 20) * 0.08) * (1 + energy * 0.5);
        const stamenHue = (hue + 60 + this.seededRandom(seed + 21) * 40) % 360;
        const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, stamenRadius);
        gradient.addColorStop(0, `hsla(${stamenHue}, ${config.saturation}%, ${85 + energy * 15}%, 0.95)`);
        gradient.addColorStop(0.6, `hsla(${(stamenHue + 30) % 360}, ${config.saturation}%, 70%, 0.5)`);
        gradient.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 50%, 0)`);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(0, 0, stamenRadius, 0, Math.PI * 2);
        ctx.fill();

        // Extra center details (seed-based)
        if (this.seededRandom(seed + 22) > 0.4) {
            const dotCount = 5 + Math.floor(this.seededRandom(seed + 23) * 4);
            for (let d = 0; d < dotCount; d++) {
                const dotAngle = (Math.PI * 2 * d) / dotCount + this.accumulatedRotation * 0.5;
                const dotDist = stamenRadius * (0.5 + this.seededRandom(seed + 24 + d) * 0.4);
                const dotSize = 2 + energy * 3;
                ctx.beginPath();
                ctx.arc(
                    Math.cos(dotAngle) * dotDist,
                    Math.sin(dotAngle) * dotDist,
                    dotSize, 0, Math.PI * 2
                );
                ctx.fillStyle = `hsla(${(stamenHue + 60) % 360}, ${config.saturation}%, 90%, ${0.7 + energy * 0.3})`;
                ctx.fill();
            }
        }

        ctx.restore();
    }

    /**
     * Spiral style - HIGHLY REACTIVE fractal spirals with dynamic pulsing
     */
    renderSpiralStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const seed = config.shapeSeed;

        ctx.save();
        ctx.translate(centerX, centerY);

        const arms = config.mirrors;
        const maxRadius = Math.min(this.canvas.width, this.canvas.height) * 0.48;

        // LAYER 0: Deep background spirals - slow, ghostly
        ctx.save();
        ctx.rotate(-this.accumulatedRotation * 0.3);
        this.drawFractalLayer(ctx, arms, maxRadius, hue, seed, 0, energy, harmonic, brightness);
        ctx.restore();

        // LAYER 1: Main spiral arms - medium speed, full reactivity
        ctx.save();
        ctx.rotate(this.accumulatedRotation * (0.8 + harmonic * 0.4));
        this.drawFractalLayer(ctx, arms, maxRadius * (0.8 + energy * 0.3), (hue + 30) % 360, seed + 100, 1, energy, harmonic, brightness);
        ctx.restore();

        // LAYER 2: Counter-rotating spirals - creates depth
        ctx.save();
        ctx.rotate(-this.accumulatedRotation * (1.2 + energy * 0.5));
        this.drawFractalLayer(ctx, arms + 2, maxRadius * (0.6 + harmonic * 0.2), (hue + 60) % 360, seed + 200, 2, energy, harmonic, brightness);
        ctx.restore();

        // LAYER 3: Fast inner spirals - very reactive
        ctx.save();
        ctx.rotate(this.accumulatedRotation * (2 + energy * 1));
        this.drawFractalLayer(ctx, arms * 2, maxRadius * (0.35 + energy * 0.15), (hue + 120) % 360, seed + 300, 3, energy, harmonic, brightness);
        ctx.restore();

        // Fractal sub-spirals at branch points
        this.drawFractalBranches(ctx, arms, maxRadius, hue, seed, energy, harmonic, brightness);

        // Reactive central vortex
        this.drawSpiralVortex(ctx, maxRadius * 0.25, hue, energy, harmonic, brightness);

        ctx.restore();
    }

    drawFractalLayer(ctx, arms, maxRadius, hue, seed, layer, energy, harmonic, brightness) {
        const config = this.config;

        // Spiral parameters that react to audio
        const spiralTurns = 2 + harmonic * 2 + brightness; // More turns with harmonic content
        const pointsPerArm = 60 + Math.floor(brightness * 40);
        const armWidth = (3 + energy * 8) * (1 - layer * 0.15); // PULSES with energy

        for (let arm = 0; arm < arms; arm++) {
            const armSeed = seed + arm * 37.7;
            const armOffset = (Math.PI * 2 * arm) / arms;
            const armHue = (hue + arm * (120 / arms) + brightness * 30) % 360;

            // MAIN SPIRAL PATH - golden ratio inspired
            ctx.beginPath();

            for (let i = 0; i < pointsPerArm; i++) {
                const t = i / pointsPerArm;

                // Spiral equation with audio modulation
                const angle = armOffset + t * Math.PI * 2 * spiralTurns;
                const baseRadius = t * maxRadius;

                // Add pulsing wave that travels outward
                const pulseWave = Math.sin(t * Math.PI * 4 - this.accumulatedRotation * 3) * energy * 20;

                // Breathing modulation
                const breathe = Math.sin(this.accumulatedRotation * 2 + t * Math.PI) * harmonic * 15;

                const spiralRadius = baseRadius * (0.5 + harmonic * 0.5) + pulseWave + breathe;

                const x = spiralRadius * Math.cos(angle);
                const y = spiralRadius * Math.sin(angle);

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            // Gradient stroke for depth
            const alpha = (0.4 + energy * 0.5) * (1 - layer * 0.15);
            ctx.strokeStyle = `hsla(${armHue}, ${config.saturation}%, ${55 + energy * 25}%, ${alpha})`;
            ctx.lineWidth = armWidth;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.stroke();

            // REACTIVE NODES along spiral - pulse with beat
            const nodeCount = 8 + Math.floor(brightness * 6);
            for (let n = 0; n < nodeCount; n++) {
                const t = (n + 1) / (nodeCount + 1);
                const angle = armOffset + t * Math.PI * 2 * spiralTurns;
                const baseRadius = t * maxRadius;
                const pulseWave = Math.sin(t * Math.PI * 4 - this.accumulatedRotation * 3) * energy * 20;
                const breathe = Math.sin(this.accumulatedRotation * 2 + t * Math.PI) * harmonic * 15;
                const spiralRadius = baseRadius * (0.5 + harmonic * 0.5) + pulseWave + breathe;

                const x = spiralRadius * Math.cos(angle);
                const y = spiralRadius * Math.sin(angle);

                // Node size EXPLODES with energy
                const baseNodeSize = 4 + this.seededRandom(armSeed + n * 7) * 8;
                const nodeSize = baseNodeSize * (0.5 + energy * 1.5 + brightness * 0.3);
                const nodeHue = (armHue + n * 15 + harmonic * 40) % 360;

                // Outer glow
                ctx.beginPath();
                ctx.arc(x, y, nodeSize * (2 + energy), 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${nodeHue}, ${config.saturation}%, 70%, ${0.1 + energy * 0.2})`;
                ctx.fill();

                // Inner bright core
                ctx.beginPath();
                ctx.arc(x, y, nodeSize, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${nodeHue}, ${config.saturation}%, ${75 + energy * 20}%, ${0.6 + energy * 0.4})`;
                ctx.fill();
            }
        }
    }

    drawFractalBranches(ctx, arms, maxRadius, hue, seed, energy, harmonic, brightness) {
        const config = this.config;

        // Branch spirals at multiple depths - FRACTAL recursion feel
        const branchDepths = [0.3, 0.5, 0.7]; // Positions along main spiral

        for (let arm = 0; arm < arms; arm++) {
            const armOffset = (Math.PI * 2 * arm) / arms + this.accumulatedRotation * (0.8 + harmonic * 0.4);

            for (let d = 0; d < branchDepths.length; d++) {
                const t = branchDepths[d];
                const branchSeed = seed + arm * 50 + d * 17;

                // Position on main spiral
                const angle = armOffset + t * Math.PI * 2 * (2 + harmonic * 2);
                const radius = t * maxRadius * (0.5 + harmonic * 0.5);
                const x = radius * Math.cos(angle);
                const y = radius * Math.sin(angle);

                // Branch parameters - REACTIVE
                const branchLength = maxRadius * 0.25 * (0.6 + energy * 0.6);
                const branchArms = 2 + Math.floor(brightness * 2);
                const branchTurns = 1 + harmonic;

                ctx.save();
                ctx.translate(x, y);
                ctx.rotate(angle + Math.PI / 2 + this.seededRandom(branchSeed) * Math.PI * 0.5);

                // Draw mini spiral branches
                for (let b = 0; b < branchArms; b++) {
                    const bAngleOffset = (Math.PI * 2 * b) / branchArms;
                    const bHue = (hue + 60 + b * 30 + d * 20) % 360;

                    ctx.beginPath();
                    for (let i = 0; i < 20; i++) {
                        const bt = i / 20;
                        const bAngle = bAngleOffset + bt * Math.PI * 2 * branchTurns;
                        const bRadius = bt * branchLength * (0.7 + energy * 0.5);

                        const bx = bRadius * Math.cos(bAngle);
                        const by = bRadius * Math.sin(bAngle);

                        if (i === 0) ctx.moveTo(bx, by);
                        else ctx.lineTo(bx, by);
                    }

                    const alpha = 0.3 + energy * 0.4;
                    ctx.strokeStyle = `hsla(${bHue}, ${config.saturation * 0.9}%, ${60 + energy * 20}%, ${alpha})`;
                    ctx.lineWidth = 1.5 + energy * 3;
                    ctx.stroke();

                    // Tiny nodes on branches
                    for (let n = 0; n < 3; n++) {
                        const bt = (n + 1) / 4;
                        const bAngle = bAngleOffset + bt * Math.PI * 2 * branchTurns;
                        const bRadius = bt * branchLength * (0.7 + energy * 0.5);
                        const bx = bRadius * Math.cos(bAngle);
                        const by = bRadius * Math.sin(bAngle);

                        ctx.beginPath();
                        ctx.arc(bx, by, 2 + energy * 4, 0, Math.PI * 2);
                        ctx.fillStyle = `hsla(${bHue}, ${config.saturation}%, 80%, ${0.5 + energy * 0.4})`;
                        ctx.fill();
                    }
                }

                // Sub-sub branches (3rd level fractal)
                if (d < 2) {
                    const subLength = branchLength * 0.4;
                    for (let s = 0; s < 3; s++) {
                        const st = 0.3 + s * 0.25;
                        const sAngle = st * Math.PI * 2 * branchTurns;
                        const sRadius = st * branchLength * (0.7 + energy * 0.5);
                        const sx = sRadius * Math.cos(sAngle);
                        const sy = sRadius * Math.sin(sAngle);

                        ctx.beginPath();
                        for (let i = 0; i < 8; i++) {
                            const sst = i / 8;
                            const ssAngle = sst * Math.PI * 1.5;
                            const ssRadius = sst * subLength * (0.5 + energy * 0.8);
                            const ssx = sx + ssRadius * Math.cos(ssAngle + sAngle);
                            const ssy = sy + ssRadius * Math.sin(ssAngle + sAngle);

                            if (i === 0) ctx.moveTo(ssx, ssy);
                            else ctx.lineTo(ssx, ssy);
                        }

                        ctx.strokeStyle = `hsla(${(hue + 90) % 360}, ${config.saturation * 0.8}%, 55%, ${0.2 + energy * 0.3})`;
                        ctx.lineWidth = 1 + energy * 1.5;
                        ctx.stroke();
                    }
                }

                ctx.restore();
            }
        }
    }

    drawSpiralVortex(ctx, radius, hue, energy, harmonic, brightness) {
        const config = this.config;

        // Hypnotic inner vortex - SUPER reactive
        const vortexArms = 8;
        const vortexTurns = 3 + energy * 2;

        // Vortex arms - spin faster with energy
        for (let arm = 0; arm < vortexArms; arm++) {
            const armOffset = (Math.PI * 2 * arm) / vortexArms + this.accumulatedRotation * (3 + energy * 2);
            const armHue = (hue + arm * 45 + harmonic * 60) % 360;

            ctx.beginPath();
            for (let i = 0; i < 30; i++) {
                const t = i / 30;
                const angle = armOffset + t * Math.PI * 2 * vortexTurns;
                const vRadius = t * radius * (0.8 + harmonic * 0.4);

                const x = vRadius * Math.cos(angle);
                const y = vRadius * Math.sin(angle);

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            ctx.strokeStyle = `hsla(${armHue}, ${config.saturation}%, ${65 + energy * 25}%, ${0.5 + energy * 0.4})`;
            ctx.lineWidth = 2 + energy * 4;
            ctx.stroke();
        }

        // Pulsing concentric rings
        const ringCount = 5;
        for (let r = 0; r < ringCount; r++) {
            const ringT = (r + 1) / (ringCount + 1);
            const ringRadius = radius * ringT * (0.8 + energy * 0.4);
            const ringHue = (hue + r * 25 + brightness * 40) % 360;

            // Ring wobble
            ctx.beginPath();
            for (let i = 0; i <= 60; i++) {
                const angle = (Math.PI * 2 * i) / 60;
                const wobble = Math.sin(angle * 6 + this.accumulatedRotation * 4) * energy * 5;
                const rr = ringRadius + wobble;
                const x = rr * Math.cos(angle);
                const y = rr * Math.sin(angle);

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            ctx.strokeStyle = `hsla(${ringHue}, ${config.saturation}%, ${70 + energy * 20}%, ${0.4 + energy * 0.3})`;
            ctx.lineWidth = 1.5 + energy * 2;
            ctx.stroke();
        }

        // Central eye - pulses dramatically
        const eyeRadius = radius * 0.3 * (1 + energy * 0.6);

        // Outer glow
        const glowGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, eyeRadius * 2);
        glowGrad.addColorStop(0, `hsla(${hue}, ${config.saturation}%, ${85 + energy * 15}%, ${0.8 + energy * 0.2})`);
        glowGrad.addColorStop(0.4, `hsla(${hue}, ${config.saturation}%, 70%, ${0.4 + energy * 0.3})`);
        glowGrad.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 50%, 0)`);
        ctx.fillStyle = glowGrad;
        ctx.beginPath();
        ctx.arc(0, 0, eyeRadius * 2, 0, Math.PI * 2);
        ctx.fill();

        // Smooth pulsing core (no facets)
        const coreGrad1 = ctx.createRadialGradient(0, 0, 0, 0, 0, eyeRadius);
        coreGrad1.addColorStop(0, `hsla(${hue}, ${config.saturation}%, ${90 + energy * 10}%, ${0.9 + energy * 0.1})`);
        coreGrad1.addColorStop(0.5, `hsla(${(hue + 30) % 360}, ${config.saturation}%, ${70 + energy * 15}%, ${0.6 + energy * 0.3})`);
        coreGrad1.addColorStop(1, `hsla(${(hue + 60) % 360}, ${config.saturation}%, 50%, ${0.2 + energy * 0.2})`);
        ctx.fillStyle = coreGrad1;
        ctx.beginPath();
        ctx.arc(0, 0, eyeRadius, 0, Math.PI * 2);
        ctx.fill();

        // White hot center
        const coreGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, eyeRadius * 0.4);
        coreGrad.addColorStop(0, `hsla(${hue}, 30%, ${95 + energy * 5}%, 1)`);
        coreGrad.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 80%, ${0.5 + energy * 0.5})`);
        ctx.fillStyle = coreGrad;
        ctx.beginPath();
        ctx.arc(0, 0, eyeRadius * 0.4 * (1 + energy * 0.3), 0, Math.PI * 2);
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
