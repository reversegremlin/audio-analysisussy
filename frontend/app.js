/**
 * Chromascope Studio - Frontend Application
 * Audio-reactive visualization controller
 */

// Per-style tooltip descriptions for each knob
const KNOB_TOOLTIPS = {
    mirrors: {
        geometric: "Number of mirror segments",
        glass: "Number of mirror wedges",
        flower: "Petal count per ring",
        spiral: "Number of spiral arms",
        circuit: "Radial symmetry segments",
        fibonacci: "Kaleidoscope mirror count",
        dmt: "Symmetry segment count",
        sacred: "Mandala fold symmetry",
        mycelial: "Branching radial arms",
        fluid: "Flow symmetry axes",
        orrery: "Decorative rings and tick marks",
        quark: "Mirror segments for fields",
    },
    baseRadius: {
        geometric: "Overall pattern size",
        glass: "Gem field radius",
        flower: "Flower bloom size",
        spiral: "Spiral reach distance",
        circuit: "Circuit grid extent",
        fibonacci: "Golden spiral radius",
        dmt: "Hyperspace field size",
        sacred: "Mandala outer radius",
        mycelial: "Network growth radius",
        fluid: "Fluid field extent",
        orrery: "Planet count and detail level",
        quark: "Quantum field extent",
    },
    orbitRadius: {
        geometric: "Ring spacing and reach",
        glass: "Gem scatter distance",
        flower: "Petal layer spacing",
        spiral: "Spiral arm length",
        circuit: "Circuit ring depth",
        fibonacci: "Golden rectangle extent",
        dmt: "Hyperspace layer depth",
        sacred: "Ring orbit distance",
        mycelial: "Hyphal reach distance",
        fluid: "Flow vortex reach",
        orrery: "Zoom level of the orrery",
        quark: "Field interaction range",
    },
    rotationSpeed: {
        geometric: "Rotation speed",
        glass: "Kaleidoscope spin rate",
        flower: "Petal rotation speed",
        spiral: "Spiral spin velocity",
        circuit: "Circuit rotation rate",
        fibonacci: "Golden spiral spin",
        dmt: "Hyperspace spin speed",
        sacred: "Mandala rotation pace",
        mycelial: "Network drift speed",
        fluid: "Flow rotation rate",
        orrery: "Orbital animation speed",
        quark: "Field phase velocity",
    },
    maxScale: {
        geometric: "Beat punch intensity",
        glass: "Beat pulse strength",
        flower: "Beat bloom punch",
        spiral: "Beat expansion force",
        circuit: "Beat pulse strength",
        fibonacci: "Beat growth punch",
        dmt: "Beat warp intensity",
        sacred: "Beat pulse strength",
        mycelial: "Beat growth burst",
        fluid: "Beat splash intensity",
        orrery: "Beat flare intensity",
        quark: "Beat field disruption",
    },
    trailAlpha: {
        geometric: "Motion trail persistence",
        glass: "Light trail smearing",
        flower: "Petal trail ghosting",
        spiral: "Spiral trail echo",
        circuit: "Trace afterglow length",
        fibonacci: "Pattern trail decay",
        dmt: "Hyperspace trail blur",
        sacred: "Mandala trail echo",
        mycelial: "Network trail fade",
        fluid: "Flow trail smear",
        orrery: "Orbital motion trails",
        quark: "Field trail persistence",
    },
    minSides: {
        geometric: "Minimum polygon sides",
        glass: "Minimum gem facets",
        flower: "Min petal layers",
        spiral: "Min node polygon sides",
        circuit: "Min circuit node sides",
        fibonacci: "Min phyllotaxis facets",
        dmt: "Min geometry complexity",
        sacred: "Min sacred geometry sides",
        mycelial: "Min node polygon sides",
        fluid: "Min vortex facets",
        orrery: "Min tick mark complexity",
        quark: "Min quark count",
    },
    maxSides: {
        geometric: "Maximum polygon sides",
        glass: "Maximum gem facets",
        flower: "Max petal layers",
        spiral: "Max node polygon sides",
        circuit: "Max circuit node sides",
        fibonacci: "Max phyllotaxis facets",
        dmt: "Max geometry complexity",
        sacred: "Max sacred geometry sides",
        mycelial: "Max node polygon sides",
        fluid: "Max vortex facets",
        orrery: "Max tick mark complexity",
        quark: "Max quark count",
    },
    baseThickness: {
        geometric: "Base line thickness",
        glass: "Base gem stroke weight",
        flower: "Base petal stroke width",
        spiral: "Base spiral line weight",
        circuit: "Base trace thickness",
        fibonacci: "Base stroke weight",
        dmt: "Base geometry stroke",
        sacred: "Base mandala line width",
        mycelial: "Base hyphal thickness",
        fluid: "Base flow line weight",
        orrery: "Base brass line weight",
        quark: "Base field line thickness",
    },
    maxThickness: {
        geometric: "Max line thickness on beats",
        glass: "Max gem stroke on beats",
        flower: "Max petal stroke on beats",
        spiral: "Max spiral width on beats",
        circuit: "Max trace width on beats",
        fibonacci: "Max stroke on beats",
        dmt: "Max geometry stroke on beats",
        sacred: "Max mandala width on beats",
        mycelial: "Max hyphal width on beats",
        fluid: "Max flow width on beats",
        orrery: "Max brass width on beats",
        quark: "Max field width on beats",
    },
};

class KaleidoscopeStudio {
    constructor() {
        // Configuration state
        this.config = {
            // Style
            style: 'geometric', // geometric, glass, flower, spiral, circuit, fibonacci, dmt, sacred, mycelial, fluid, orrery, quark
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
            fps: 60,
            quality: 'high'
        };

        // Shared style presets loaded from the backend (styles.json)
        this.stylePresets = null;

        // High-level session model (audio, style, mapping, export)
        this.session = this.createDefaultSession();

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
        this.lastFrameTime = performance.now();

        // Canvas
        this.canvas = document.getElementById('visualizerCanvas');
        this.ctx = this.canvas.getContext('2d', { alpha: false }); // Opaque for performance
        this.waveformCanvas = document.getElementById('waveformCanvas');
        this.waveformCtx = this.waveformCanvas.getContext('2d');

        // Set canvas size from config
        this.canvas.width = this.config.width;
        this.canvas.height = this.config.height;

        // Create shared render engine (config passed by reference — slider changes propagate)
        this.renderer = new ChromascopeRenderer(this.config, this.canvas);

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupKnobs();
        this.updateKnobTooltips();
        // Load shared style presets in the background
        this.loadStylePresets();
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

        // CLI path fields
        const cliInputSong = document.getElementById('cliInputSong');
        const cliOutputFolder = document.getElementById('cliOutputFolder');
        if (cliInputSong) {
            cliInputSong.addEventListener('input', () => {
                if (!cliInputSong.value.trim()) {
                    cliInputSong.placeholder = '/path/to/your song.mp3';
                }
            });
        }
        if (cliOutputFolder) {
            cliOutputFolder.addEventListener('blur', () => {
                if (!cliOutputFolder.value.trim()) {
                    cliOutputFolder.value = './renders';
                }
            });
        }

        // Style buttons
        const glassSlicesControl = document.getElementById('glassSlicesControl');
        document.querySelectorAll('.style-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.config.style = btn.dataset.style;
                this.applyStylePreset(btn.dataset.style);
                this.updateKnobTooltips();
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
        const customResPanel = document.getElementById('customResolution');
        document.querySelectorAll('[data-resolution]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-resolution]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');

                if (e.target.dataset.resolution === 'custom') {
                    customResPanel.style.display = 'flex';
                    this.applyCustomResolution();
                } else {
                    customResPanel.style.display = 'none';
                    const [w, h] = e.target.dataset.resolution.split('x').map(Number);
                    this.config.width = w;
                    this.config.height = h;
                    this.canvas.width = w;
                    this.canvas.height = h;
                    document.getElementById('resolutionBadge').textContent = `${w} × ${h}`;
                }
            });
        });

        // Custom resolution inputs
        document.getElementById('customWidth').addEventListener('input', () => this.applyCustomResolution());
        document.getElementById('customHeight').addEventListener('input', () => this.applyCustomResolution());

        // FPS buttons
        document.querySelectorAll('[data-fps]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-fps]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.config.fps = parseInt(e.target.dataset.fps);
                document.getElementById('fpsBadge').textContent = `${this.config.fps} FPS`;
            });
        });

        // Quality buttons
        document.querySelectorAll('[data-quality]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-quality]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.config.quality = e.target.dataset.quality;
            });
        });

        // Export button
        document.getElementById('exportBtn').addEventListener('click', () => this.exportVideo());

        // CLI command export
        document.getElementById('copyCliBtn').addEventListener('click', () => this.showCliCommand());
        document.getElementById('cliCopyIcon').addEventListener('click', () => this.copyCliToClipboard());
        document.getElementById('exportConfigBtn').addEventListener('click', () => this.exportConfigJson());

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

    updateKnobTooltips() {
        const style = this.config.style;
        document.querySelectorAll('.knob-control').forEach(control => {
            const knob = control.querySelector('.knob');
            if (!knob) return;
            const param = knob.dataset.param;
            if (!param || !KNOB_TOOLTIPS[param]) return;
            const tip = KNOB_TOOLTIPS[param][style] || '';
            if (tip) {
                control.setAttribute('data-tooltip', tip);
            } else {
                control.removeAttribute('data-tooltip');
            }
        });
    }

    createDefaultSession() {
        /**
         * Create a default session snapshot that captures the current
         * configuration in a JSON-serializable structure.
         */
        return {
            style: this.config.style,
            geometry: {
                mirrors: this.config.mirrors,
                baseRadius: this.config.baseRadius,
                orbitRadius: this.config.orbitRadius,
                rotationSpeed: this.config.rotationSpeed,
                minSides: this.config.minSides,
                maxSides: this.config.maxSides,
                baseThickness: this.config.baseThickness,
                maxThickness: this.config.maxThickness
            },
            dynamics: {
                maxScale: this.config.maxScale,
                trailAlpha: this.config.trailAlpha,
                attackMs: this.config.attackMs,
                releaseMs: this.config.releaseMs
            },
            colors: {
                bgColor: this.config.bgColor,
                bgColor2: this.config.bgColor2,
                accentColor: this.config.accentColor,
                chromaColors: this.config.chromaColors,
                saturation: this.config.saturation
            },
            background: {
                dynamicBg: this.config.dynamicBg,
                bgReactivity: this.config.bgReactivity,
                bgParticles: this.config.bgParticles,
                bgPulse: this.config.bgPulse
            },
            export: {
                width: this.config.width,
                height: this.config.height,
                fps: this.config.fps
            }
        };
    }

    getSessionSnapshot() {
        /**
         * Return a fresh snapshot of the current session state.
         */
        const snapshot = this.createDefaultSession();
        snapshot.style = this.config.style;
        return snapshot;
    }

    applySessionSnapshot(snapshot) {
        /**
         * Apply a previously saved session snapshot back onto the current
         * configuration. Unknown fields are ignored.
         */
        if (!snapshot || typeof snapshot !== 'object') return;

        if (snapshot.style) {
            this.config.style = snapshot.style;
            this.applyStylePreset(snapshot.style);
        }

        const g = snapshot.geometry || {};
        if (typeof g.mirrors === 'number') this.config.mirrors = g.mirrors;
        if (typeof g.baseRadius === 'number') this.config.baseRadius = g.baseRadius;
        if (typeof g.orbitRadius === 'number') this.config.orbitRadius = g.orbitRadius;
        if (typeof g.rotationSpeed === 'number') this.config.rotationSpeed = g.rotationSpeed;
        if (typeof g.minSides === 'number') this.config.minSides = g.minSides;
        if (typeof g.maxSides === 'number') this.config.maxSides = g.maxSides;
        if (typeof g.baseThickness === 'number') this.config.baseThickness = g.baseThickness;
        if (typeof g.maxThickness === 'number') this.config.maxThickness = g.maxThickness;

        const d = snapshot.dynamics || {};
        if (typeof d.maxScale === 'number') this.config.maxScale = d.maxScale;
        if (typeof d.trailAlpha === 'number') this.config.trailAlpha = d.trailAlpha;
        if (typeof d.attackMs === 'number') this.config.attackMs = d.attackMs;
        if (typeof d.releaseMs === 'number') this.config.releaseMs = d.releaseMs;

        const c = snapshot.colors || {};
        if (typeof c.bgColor === 'string') this.config.bgColor = c.bgColor;
        if (typeof c.bgColor2 === 'string') this.config.bgColor2 = c.bgColor2;
        if (typeof c.accentColor === 'string') this.config.accentColor = c.accentColor;
        if (typeof c.chromaColors === 'boolean') this.config.chromaColors = c.chromaColors;
        if (typeof c.saturation === 'number') this.config.saturation = c.saturation;

        const b = snapshot.background || {};
        if (typeof b.dynamicBg === 'boolean') this.config.dynamicBg = b.dynamicBg;
        if (typeof b.bgReactivity === 'number') this.config.bgReactivity = b.bgReactivity;
        if (typeof b.bgParticles === 'boolean') this.config.bgParticles = b.bgParticles;
        if (typeof b.bgPulse === 'boolean') this.config.bgPulse = b.bgPulse;

        const ex = snapshot.export || {};
        if (typeof ex.width === 'number') this.config.width = ex.width;
        if (typeof ex.height === 'number') this.config.height = ex.height;
        if (typeof ex.fps === 'number') this.config.fps = ex.fps;

        // After applying, refresh tooltips and session cache
        this.updateKnobTooltips();
        this.session = this.getSessionSnapshot();
    }

    saveSessionToLocalStorage(key = 'chromascope_session') {
        /**
         * Persist the current session into localStorage under a given key.
         */
        try {
            const snapshot = this.getSessionSnapshot();
            window.localStorage.setItem(key, JSON.stringify(snapshot));
        } catch (e) {
            console.warn('Failed to save session', e);
        }
    }

    loadSessionFromLocalStorage(key = 'chromascope_session') {
        /**
         * Load a session from localStorage and apply it if present.
         */
        try {
            const raw = window.localStorage.getItem(key);
            if (!raw) return;
            const snapshot = JSON.parse(raw);
            this.applySessionSnapshot(snapshot);
        } catch (e) {
            console.warn('Failed to load session', e);
        }
    }

    async loadStylePresets() {
        /**
         * Load shared style presets from the backend so that Studio and the
         * Python renderer use the same style definitions.
         */
        try {
            const response = await fetch('/styles.json');
            if (!response.ok) {
                return;
            }
            const data = await response.json();
            this.stylePresets = data.kaleidoscope || null;
        } catch (err) {
            // Non-fatal: fall back to built-in defaults if presets can't load
            console.warn('Failed to load style presets', err);
        }
    }

    applyStylePreset(styleName) {
        /**
         * Apply a style preset from the shared preset table to the current
         * Studio config. Only touches geometry/dynamics parameters.
         */
        if (!this.stylePresets) return;
        const preset = this.stylePresets[styleName];
        if (!preset) return;

        const k = preset;

        if (typeof k.num_mirrors === 'number') this.config.mirrors = k.num_mirrors;
        if (typeof k.base_radius === 'number') this.config.baseRadius = k.base_radius;
        if (typeof k.orbit_radius === 'number') this.config.orbitRadius = k.orbit_radius;
        if (typeof k.rotation_speed === 'number') this.config.rotationSpeed = k.rotation_speed;
        if (typeof k.max_scale === 'number') this.config.maxScale = k.max_scale;
        if (typeof k.trail_alpha === 'number') this.config.trailAlpha = k.trail_alpha;
        if (typeof k.min_sides === 'number') this.config.minSides = k.min_sides;
        if (typeof k.max_sides === 'number') this.config.maxSides = k.max_sides;
        if (typeof k.base_thickness === 'number') this.config.baseThickness = k.base_thickness;
        if (typeof k.max_thickness === 'number') this.config.maxThickness = k.max_thickness;
    }

    async loadAudioFile(file) {
        this.audioFile = file; // Store for server-side export

        const cliInputSong = document.getElementById('cliInputSong');
        if (cliInputSong && (!cliInputSong.value || cliInputSong.value === '/path/to/your song.mp3')) {
            cliInputSong.value = file.name;
        }

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
            this.renderer.renderFrame(frameData, deltaTime, this.isPlaying);

            requestAnimationFrame(animate);
        };

        requestAnimationFrame(animate);
    }


    render() {
        // Initial render with idle state
        this.lastFrameTime = performance.now();
        this.renderer.renderFrame(null, 16.67, this.isPlaying);
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

    applyCustomResolution() {
        const w = parseInt(document.getElementById('customWidth').value) || 1280;
        const h = parseInt(document.getElementById('customHeight').value) || 720;
        // Clamp to reasonable bounds
        const width = Math.max(320, Math.min(7680, w));
        const height = Math.max(240, Math.min(4320, h));
        this.config.width = width;
        this.config.height = height;
        this.canvas.width = width;
        this.canvas.height = height;
        document.getElementById('resolutionBadge').textContent = `${width} × ${height}`;
    }

    generateCliCommand() {
        const c = this.config;
        const parts = ['python -m chromascope.render_video'];
        const cliInputSong = document.getElementById('cliInputSong')?.value?.trim();
        const cliOutputFolder = document.getElementById('cliOutputFolder')?.value?.trim() || './renders';

        // Audio input path
        const inputSongPath = cliInputSong || this.audioFile?.name || '/path/to/your song.mp3';
        parts.push(this.shellQuote(inputSongPath));

        // Output
        const outputFileName = `chromascope_${c.style}_export.mp4`;
        const normalizedOutputFolder = cliOutputFolder.replace(/[\\/]+$/, '');
        const outputPath = normalizedOutputFolder ? `${normalizedOutputFolder}/${outputFileName}` : outputFileName;
        parts.push(`-o ${this.shellQuote(outputPath)}`);

        // Resolution & FPS
        parts.push(`--width ${c.width}`);
        parts.push(`--height ${c.height}`);
        parts.push(`--fps ${c.fps}`);

        // Style
        if (c.style !== 'geometric') {
            parts.push(`--style ${c.style}`);
        }

        // Quality
        if (c.quality !== 'high') {
            parts.push(`--quality ${c.quality}`);
        }

        // Geometry params (only if non-default)
        if (c.mirrors !== 8) parts.push(`--mirrors ${c.mirrors}`);
        if (c.trailAlpha !== 40) parts.push(`--trail ${c.trailAlpha}`);
        if (c.baseRadius !== 150) parts.push(`--base-radius ${c.baseRadius}`);
        if (c.orbitRadius !== 200) parts.push(`--orbit-radius ${c.orbitRadius}`);
        if (c.rotationSpeed !== 2.0) parts.push(`--rotation-speed ${c.rotationSpeed}`);
        if (c.maxScale !== 1.8) parts.push(`--max-scale ${c.maxScale}`);
        if (c.minSides !== 3) parts.push(`--min-sides ${c.minSides}`);
        if (c.maxSides !== 12) parts.push(`--max-sides ${c.maxSides}`);
        if (c.baseThickness !== 3) parts.push(`--base-thickness ${c.baseThickness}`);
        if (c.maxThickness !== 12) parts.push(`--max-thickness ${c.maxThickness}`);

        // Colors
        if (c.bgColor !== '#05050f') parts.push(`--bg-color "${c.bgColor}"`);
        if (c.bgColor2 !== '#1a0a2e') parts.push(`--bg-color2 "${c.bgColor2}"`);
        if (c.accentColor !== '#f59e0b') parts.push(`--accent-color "${c.accentColor}"`);
        if (c.saturation !== 85) parts.push(`--saturation ${c.saturation}`);
        if (!c.chromaColors) parts.push('--no-chroma-colors');

        // Background effects
        if (!c.dynamicBg) parts.push('--no-dynamic-bg');
        if (!c.bgParticles) parts.push('--no-particles');
        if (!c.bgPulse) parts.push('--no-pulse');
        if (c.bgReactivity !== 70) parts.push(`--bg-reactivity ${c.bgReactivity}`);

        // Format with line continuation for readability
        if (parts.length <= 4) {
            return parts.join(' ');
        }
        return parts[0] + ' ' + parts[1] + ' \\\n    ' +
            parts.slice(2).join(' \\\n    ');
    }

    shellQuote(value) {
        const text = String(value ?? '');
        if (!text) return "''";
        return `'${text.replace(/'/g, `'\\''`)}'`;
    }

    showCliCommand() {
        const output = document.getElementById('cliCommandOutput');
        const text = document.getElementById('cliCommandText');
        const hint = document.getElementById('cliCommandHint');

        const command = this.generateCliCommand();
        text.textContent = command;
        output.style.display = 'block';
        hint.textContent = 'Paste into your terminal to render offline';
        hint.classList.remove('copied');
    }

    async copyCliToClipboard() {
        const text = document.getElementById('cliCommandText').textContent;
        const hint = document.getElementById('cliCommandHint');
        try {
            await navigator.clipboard.writeText(text);
            hint.textContent = 'Copied to clipboard!';
            hint.classList.add('copied');
            setTimeout(() => {
                hint.textContent = 'Paste into your terminal to render offline';
                hint.classList.remove('copied');
            }, 2000);
        } catch (err) {
            // Fallback: select text for manual copy
            const range = document.createRange();
            range.selectNodeContents(document.getElementById('cliCommandText'));
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            hint.textContent = 'Press Ctrl+C to copy';
        }
    }

    exportConfigJson() {
        const snapshot = this.getSessionSnapshot();
        // Flatten into the config format the CLI expects
        const flatConfig = {
            style: this.config.style,
            width: this.config.width,
            height: this.config.height,
            fps: this.config.fps,
            quality: this.config.quality,
            mirrors: this.config.mirrors,
            baseRadius: this.config.baseRadius,
            orbitRadius: this.config.orbitRadius,
            rotationSpeed: this.config.rotationSpeed,
            maxScale: this.config.maxScale,
            trailAlpha: this.config.trailAlpha,
            minSides: this.config.minSides,
            maxSides: this.config.maxSides,
            baseThickness: this.config.baseThickness,
            maxThickness: this.config.maxThickness,
            bgColor: this.config.bgColor,
            bgColor2: this.config.bgColor2,
            accentColor: this.config.accentColor,
            chromaColors: this.config.chromaColors,
            saturation: this.config.saturation,
            dynamicBg: this.config.dynamicBg,
            bgReactivity: this.config.bgReactivity,
            bgParticles: this.config.bgParticles,
            bgPulse: this.config.bgPulse,
            shapeSeed: this.config.shapeSeed,
            glassSlices: this.config.glassSlices,
        };

        const json = JSON.stringify(flatConfig, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chromascope_${this.config.style}_config.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    async exportVideo() {
        if (!this.audioFile) {
            alert('Please load an audio file first');
            return;
        }

        const exportBtn = document.getElementById('exportBtn');
        const exportProgress = document.getElementById('exportProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');

        exportBtn.disabled = true;
        exportProgress.style.display = 'block';

        let pollTimer = null;

        try {
            progressText.textContent = 'Uploading audio...';
            progressFill.style.width = '0%';

            // Build form data with audio file + full config
            const formData = new FormData();
            formData.append('audio', this.audioFile);
            formData.append('config', JSON.stringify(this.config));

            // POST to server render endpoint
            const startResp = await fetch('/api/render', {
                method: 'POST',
                body: formData,
            });

            if (!startResp.ok) {
                throw new Error(`Server error: ${startResp.status} ${startResp.statusText}`);
            }

            const { task_id } = await startResp.json();
            progressText.textContent = 'Rendering on server...';

            // Poll for progress
            const pollStatus = async () => {
                try {
                    const resp = await fetch(`/api/render/status/${task_id}`);
                    if (!resp.ok) return;
                    const status = await resp.json();

                    progressFill.style.width = `${status.progress}%`;
                    progressText.textContent = status.message || `Rendering... ${status.progress}%`;

                    if (status.error) {
                        throw new Error(status.error);
                    }

                    if (status.complete) {
                        clearInterval(pollTimer);
                        pollTimer = null;

                        // Download the rendered video
                        progressText.textContent = 'Downloading MP4...';
                        const dlResp = await fetch(`/api/render/download/${task_id}`);
                        if (!dlResp.ok) throw new Error('Download failed');

                        const blob = await dlResp.blob();
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `chromascope_${this.config.style}_export.mp4`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);

                        progressFill.style.width = '100%';
                        progressText.textContent = 'Export complete!';

                        setTimeout(() => {
                            exportBtn.disabled = false;
                            exportProgress.style.display = 'none';
                            progressFill.style.width = '0%';
                        }, 3000);
                    }
                } catch (err) {
                    clearInterval(pollTimer);
                    pollTimer = null;
                    console.error('Export error:', err);
                    progressText.textContent = 'Export failed: ' + err.message;
                    setTimeout(() => {
                        exportBtn.disabled = false;
                        exportProgress.style.display = 'none';
                        progressFill.style.width = '0%';
                    }, 3000);
                }
            };

            pollTimer = setInterval(pollStatus, 1000);

        } catch (error) {
            console.error('Export error:', error);
            progressText.textContent = 'Export failed: ' + error.message;
            if (pollTimer) clearInterval(pollTimer);
            setTimeout(() => {
                exportBtn.disabled = false;
                exportProgress.style.display = 'none';
                progressFill.style.width = '0%';
            }, 3000);
        }
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.studio = new KaleidoscopeStudio();
});
