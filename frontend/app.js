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
        this.manifest = null;
        this.currentFrame = 0;

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
        const bpm = 120;
        const beatsPerSecond = bpm / 60;
        const framesPerBeat = fps / beatsPerSecond;

        // Use deterministic pseudo-random for consistency
        const seededRandom = (seed) => {
            const x = Math.sin(seed * 12.9898) * 43758.5453;
            return x - Math.floor(x);
        };

        for (let i = 0; i < totalFrames; i++) {
            const time = i / fps;
            const beatPhase = (i % framesPerBeat) / framesPerBeat;
            // Only trigger beat on the first frame of each beat period
            const isBeat = (i % Math.round(framesPerBeat)) === 0;

            // Smooth energy curves using sine waves (no randomness for smoothness)
            const slowWave = Math.sin(time * 0.3) * 0.5 + 0.5;
            const medWave = Math.sin(time * 0.8 + 1) * 0.5 + 0.5;
            const fastWave = Math.sin(time * 2.1 + 2) * 0.5 + 0.5;

            // Percussive: spike on beats, smooth decay otherwise
            let percussiveImpact;
            if (isBeat) {
                percussiveImpact = 0.85 + seededRandom(i) * 0.15;
            } else {
                // Exponential decay after beat
                const decayPhase = beatPhase;
                const decay = Math.exp(-decayPhase * 8);
                percussiveImpact = 0.1 + decay * 0.6;
            }

            // Harmonic: smooth flowing energy
            const harmonicEnergy = 0.3 + slowWave * 0.25 + medWave * 0.2 + fastWave * 0.1;

            // Brightness: gentle variation
            const spectralBrightness = 0.4 + medWave * 0.3 + fastWave * 0.15;

            // Cycle through chroma (change every 2 beats for musical feel)
            const chromaNames = ['C', 'G', 'Am', 'F', 'C', 'G', 'D', 'Em'].map(c => c.charAt(0));
            const fullChromaNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
            const beatsElapsed = Math.floor(time * beatsPerSecond);
            const chromaIndex = Math.floor(beatsElapsed / 2) % fullChromaNames.length;

            frames.push({
                frame_index: i,
                time: time,
                is_beat: isBeat,
                is_onset: isBeat,
                percussive_impact: Math.min(1, Math.max(0, percussiveImpact)),
                harmonic_energy: Math.min(1, Math.max(0, harmonicEnergy)),
                global_energy: (percussiveImpact + harmonicEnergy) / 2,
                spectral_brightness: Math.min(1, Math.max(0, spectralBrightness)),
                dominant_chroma: fullChromaNames[chromaIndex]
            });
        }

        this.manifest = {
            metadata: {
                bpm: bpm,
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

            // Get current frame data
            let frameData = null;
            if (this.manifest && this.manifest.frames) {
                const frameIndex = Math.floor(currentTime * this.config.fps);
                frameData = this.manifest.frames[Math.min(frameIndex, this.manifest.frames.length - 1)];

                // Beat flash effect (with longer debounce)
                if (frameData && frameData.is_beat && this.isPlaying) {
                    if (timestamp - lastBeatTime > 400) { // At least 400ms between flashes
                        lastBeatTime = timestamp;
                        document.querySelector('.canvas-container').classList.add('beat');
                        setTimeout(() => {
                            document.querySelector('.canvas-container').classList.remove('beat');
                        }, 80);
                    }
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

        // Smooth, slow gradient blend based on harmonic energy
        const blendPhase = Math.sin(this.bgState.gradientAngle) * 0.5 + 0.5;
        const energyBlend = this.smoothedValues.harmonicEnergy * reactivity * 0.3;
        const blend = blendPhase * 0.4 + energyBlend;

        // Gentle center movement
        const centerX = width / 2 + Math.sin(this.bgState.gradientAngle * 2) * width * 0.05 * reactivity;
        const centerY = height / 2 + Math.cos(this.bgState.gradientAngle * 1.5) * height * 0.05 * reactivity;

        // Subtle pulse radius on beats
        const pulseExpand = this.bgState.pulseIntensity * 100 * reactivity;
        const baseRadius = Math.max(width, height) * 0.9;
        const outerRadius = baseRadius + pulseExpand;

        // Create gradient
        const gradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, outerRadius
        );

        // Interpolate colors smoothly
        const midColor = {
            r: Math.round(color1.r + (color2.r - color1.r) * blend),
            g: Math.round(color1.g + (color2.g - color1.g) * blend),
            b: Math.round(color1.b + (color2.b - color1.b) * blend)
        };

        // Subtle brightness boost on beats (much gentler)
        const brightnessBoost = this.bgState.pulseIntensity * 15 * reactivity;
        gradient.addColorStop(0, `rgb(${Math.min(255, midColor.r + brightnessBoost)}, ${Math.min(255, midColor.g + brightnessBoost)}, ${Math.min(255, midColor.b + brightnessBoost)})`);
        gradient.addColorStop(0.4, `rgb(${midColor.r}, ${midColor.g}, ${midColor.b})`);
        gradient.addColorStop(1, `rgb(${color1.r}, ${color1.g}, ${color1.b})`);

        // Gentle fade for trails - lower alpha = more trail persistence
        const fadeAmount = (100 - config.trailAlpha) / 100;
        ctx.globalAlpha = fadeAmount * 0.15 + 0.02;
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);

        // Subtle vignette
        const vignetteGradient = ctx.createRadialGradient(
            width / 2, height / 2, height * 0.4,
            width / 2, height / 2, Math.max(width, height) * 0.85
        );
        vignetteGradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
        vignetteGradient.addColorStop(1, 'rgba(0, 0, 0, 0.3)');
        ctx.globalAlpha = 0.2;
        ctx.fillStyle = vignetteGradient;
        ctx.fillRect(0, 0, width, height);

        ctx.globalAlpha = 1;

        // Render particles if enabled
        if (config.bgParticles) {
            this.renderParticles(deltaTime);
        }

        // Render pulse rings on beats (only when visible)
        if (config.bgPulse && this.bgState.pulseIntensity > 0.05) {
            this.renderPulseRings();
        }
    }

    renderParticles(deltaTime) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const config = this.config;

        const reactivity = config.bgReactivity / 100;
        const energyBoost = 1 + this.smoothedValues.harmonicEnergy * reactivity;

        ctx.save();

        this.bgState.particles.forEach(particle => {
            // Update position gently
            particle.x += Math.cos(particle.angle) * particle.speed * deltaTime * 0.03 * energyBoost;
            particle.y += Math.sin(particle.angle) * particle.speed * deltaTime * 0.03 * energyBoost;

            // Wrap around edges
            if (particle.x < 0) particle.x = width;
            if (particle.x > width) particle.x = 0;
            if (particle.y < 0) particle.y = height;
            if (particle.y > height) particle.y = 0;

            // Gentle pulse brightness
            particle.pulse += deltaTime * 0.002;
            const pulseBrightness = Math.sin(particle.pulse) * 0.2 + 0.8;
            const beatBrightness = 1 + this.bgState.pulseIntensity * 0.3;

            // Draw particle - subtle
            const alpha = particle.brightness * pulseBrightness * beatBrightness * reactivity * 0.6;
            const size = particle.size * (1 + this.smoothedValues.percussiveImpact * 0.3);

            ctx.beginPath();
            ctx.arc(particle.x, particle.y, size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${Math.min(0.5, alpha)})`;
            ctx.fill();

            // Subtle glow on high energy only
            if (this.smoothedValues.percussiveImpact > 0.6) {
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, size * 2.5, 0, Math.PI * 2);
                const glowAlpha = (this.smoothedValues.percussiveImpact - 0.6) * alpha * 0.15;
                ctx.fillStyle = `rgba(255, 255, 255, ${Math.min(0.2, glowAlpha)})`;
                ctx.fill();
            }
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
        const maxRadius = Math.max(width, height) * 0.5;

        // Subtle expanding rings
        const numRings = 2;
        const baseAlpha = this.bgState.pulseIntensity * 0.08;

        ctx.save();

        for (let i = 0; i < numRings; i++) {
            const phase = (1 - this.bgState.pulseIntensity + i * 0.3) % 1;
            const radius = phase * maxRadius;
            const alpha = baseAlpha * (1 - phase * phase); // Quadratic falloff

            if (alpha > 0.005 && radius > 10) {
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
                ctx.strokeStyle = config.accentColor;
                ctx.lineWidth = 1.5;
                ctx.globalAlpha = alpha;
                ctx.stroke();
            }
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

        // Smooth all incoming values for fluid motion
        const smoothFactor = Math.min(1, deltaTime * 0.008); // Time-based smoothing
        this.smoothedValues.percussiveImpact = this.lerp(
            this.smoothedValues.percussiveImpact,
            frameData.percussive_impact,
            frameData.is_beat ? 0.5 : smoothFactor // Faster response on beats
        );
        this.smoothedValues.harmonicEnergy = this.lerp(
            this.smoothedValues.harmonicEnergy,
            frameData.harmonic_energy,
            smoothFactor * 0.5
        );
        this.smoothedValues.spectralBrightness = this.lerp(
            this.smoothedValues.spectralBrightness,
            frameData.spectral_brightness,
            smoothFactor * 0.3
        );

        // Update background state
        this.bgState.gradientAngle += deltaTime * 0.00008 * (1 + this.smoothedValues.harmonicEnergy * 0.5);

        // Smoother pulse - faster decay, gentler attack
        const targetPulse = frameData.is_beat ? 0.6 : 0;
        const pulseLerp = frameData.is_beat ? 0.3 : 0.08;
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
