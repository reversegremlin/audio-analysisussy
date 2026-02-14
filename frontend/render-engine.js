/**
 * Chromascope Render Engine
 * Shared rendering module for browser and Node.js server-side export.
 * Extracted from app.js — contains all visualization styles, backgrounds,
 * helpers, and the main renderFrame() dispatch.
 *
 * UMD wrapper: works as <script> tag in browser or require() in Node.js.
 */
(function(root, factory) {
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = factory();
    } else {
        root.ChromascopeRenderer = factory();
    }
}(typeof self !== 'undefined' ? self : this, function() {

    class ChromascopeRenderer {
        constructor(config, canvas) {
            this.config = config;
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');

            // Smoothed audio-reactive values
            this.smoothedValues = {
                percussiveImpact: 0,
                harmonicEnergy: 0.3,
                spectralBrightness: 0.5
            };
            this.smoothingFactor = 0.15;

            // Rotation accumulator
            this.accumulatedRotation = 0;

            // Background animation state
            this.bgState = {
                gradientAngle: 0,
                pulseIntensity: 0,
                particles: [],
                noiseOffset: 0
            };

            // Binary star orrery simulation state
            this._binaryState = null;
            this._binaryLastRot = 0;
            this._binarySeed = -1;

            // Chroma-to-hue mapping
            this.chromaToHue = {
                'C': 0, 'C#': 30, 'D': 60, 'D#': 90, 'E': 120, 'F': 150,
                'F#': 180, 'G': 210, 'G#': 240, 'A': 270, 'A#': 300, 'B': 330
            };

            this.initParticles();
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

        if (this._bgColorBlend === undefined) {
            this._bgColorBlend = 0.3;
        }
        const targetBlend = 0.3 + this.smoothedValues.harmonicEnergy * reactivity * 0.3;
        this._bgColorBlend = this.lerp(this._bgColorBlend, targetBlend, 0.001);
        const blend = this._bgColorBlend;

        const centerX = width / 2;
        const centerY = height / 2;
        const baseRadius = Math.max(width, height) * 0.95;

        // Create radial gradient base
        const gradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, baseRadius
        );

        const midColor = {
            r: Math.round(color1.r + (color2.r - color1.r) * blend),
            g: Math.round(color1.g + (color2.g - color1.g) * blend),
            b: Math.round(color1.b + (color2.b - color1.b) * blend)
        };

        gradient.addColorStop(0, `rgb(${midColor.r}, ${midColor.g}, ${midColor.b})`);
        gradient.addColorStop(0.6, `rgb(${Math.round(midColor.r * 0.7 + color1.r * 0.3)}, ${Math.round(midColor.g * 0.7 + color1.g * 0.3)}, ${Math.round(midColor.b * 0.7 + color1.b * 0.3)})`);
        gradient.addColorStop(1, `rgb(${color1.r}, ${color1.g}, ${color1.b})`);

        const fadeAmount = (100 - config.trailAlpha) / 100;
        ctx.globalAlpha = fadeAmount * 0.06 + 0.01;
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);

        // Reset alpha before drawing fractal background
        ctx.globalAlpha = 1;

        // --- Full-screen style-specific background pattern ---
        this.renderStyledBackground(ctx, width, height, centerX, centerY, reactivity, deltaTime);

        // Vignette
        const vignetteGradient = ctx.createRadialGradient(
            centerX, centerY, height * 0.3,
            centerX, centerY, Math.max(width, height) * 0.85
        );
        vignetteGradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
        vignetteGradient.addColorStop(1, 'rgba(0, 0, 0, 0.4)');
        ctx.globalAlpha = 0.12;
        ctx.fillStyle = vignetteGradient;
        ctx.fillRect(0, 0, width, height);

        ctx.globalAlpha = 1;

        // Render particles if enabled
        if (config.bgParticles) {
            this.renderParticles(deltaTime);
        }

        // Render pulse rings on beats
        if (config.bgPulse && this.bgState.pulseIntensity > 0.1) {
            this.renderPulseRings();
        }
    }

    /**
     * Style-aware background dispatcher
     * Routes to unique background renderer per visualization style
     */
    renderStyledBackground(ctx, width, height, centerX, centerY, reactivity, deltaTime) {
        if (this._bgFractalRotation === undefined) {
            this._bgFractalRotation = 0;
        }
        const harmonic = this.smoothedValues.harmonicEnergy;
        this._bgFractalRotation += deltaTime * 0.00004 * (1 + harmonic * reactivity * 0.5);

        switch (this.config.style) {
            case 'glass':
                this.renderGlassBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'flower':
                this.renderFlowerBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'spiral':
                this.renderSpiralBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'circuit':
                this.renderCircuitBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'fibonacci':
                this.renderFibonacciBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'dmt':
                this.renderDMTBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'sacred':
                this.renderSacredBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'mycelial':
                this.renderMycelialBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'fluid':
                this.renderFluidBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'orrery':
                this.renderOrreryBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'quark':
                this.renderQuarkBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            case 'fractal':
                this.renderFractalBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
            default:
                this.renderGeometricBackground(ctx, width, height, centerX, centerY, reactivity);
                break;
        }
    }

    /**
     * Geometric background — Concentric polygon rings with radial guides
     * 3 parallax layers: polygon rings (far), radial lines (mid), floating polygons (near)
     */
    renderGeometricBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const mirrors = config.mirrors;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const baseAlpha = 0.06 + reactivity * 0.10 + energy * reactivity * 0.08;

        // --- FAR LAYER (0.3x): Concentric polygon rings ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.3);

        const ringCount = 7;
        for (let r = 0; r < ringCount; r++) {
            const ringRadius = maxDim * (0.2 + r * 0.18) * (0.9 + energy * reactivity * 0.1);
            const ringSides = mirrors;
            const ringHue = (bgHue + r * 25 + brightness * 15) % 360;

            ctx.beginPath();
            for (let i = 0; i <= ringSides; i++) {
                const angle = (Math.PI * 2 * i) / ringSides;
                const wobble = Math.sin(angle * 3 + rot * 5) * maxDim * 0.008 * energy * reactivity;
                const px = Math.cos(angle) * (ringRadius + wobble);
                const py = Math.sin(angle) * (ringRadius + wobble);
                i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
            }
            ctx.strokeStyle = `hsla(${ringHue}, ${config.saturation * 0.5}%, 55%, ${baseAlpha * (1 - r * 0.08)})`;
            ctx.lineWidth = 0.8 + energy * reactivity * 1.5;
            ctx.stroke();
        }
        ctx.restore();

        // --- MID LAYER (0.7x, counter-rotate): Radial guide lines ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.7);

        const guideCount = mirrors * 3;
        for (let i = 0; i < guideCount; i++) {
            const angle = (Math.PI * 2 * i) / guideCount;
            const lineHue = (bgHue + i * (120 / guideCount) + harmonic * 20) % 360;
            const breathe = 0.9 + Math.sin(rot * 3 + i * 0.5) * 0.1 * reactivity;

            ctx.beginPath();
            ctx.moveTo(
                Math.cos(angle) * maxDim * 0.1,
                Math.sin(angle) * maxDim * 0.1
            );
            ctx.lineTo(
                Math.cos(angle) * maxDim * 1.3 * breathe,
                Math.sin(angle) * maxDim * 1.3 * breathe
            );

            if (i % 2 === 0) {
                ctx.setLineDash([8, 12]);
            } else {
                ctx.setLineDash([]);
            }
            ctx.strokeStyle = `hsla(${lineHue}, ${config.saturation * 0.4}%, 55%, ${baseAlpha * 0.7})`;
            ctx.lineWidth = 0.5 + energy * reactivity * 1.5;
            ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.restore();

        // --- NEAR LAYER (1.2x): Small floating polygon outlines ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 1.2);

        const floatCount = 18;
        for (let i = 0; i < floatCount; i++) {
            const seed = i * 7 + 42;
            const dist = maxDim * (0.25 + this.seededRandom(seed) * 0.85);
            const baseAngle = this.seededRandom(seed + 1) * Math.PI * 2;
            const sides = 3 + Math.floor(this.seededRandom(seed + 2) * 4);
            const size = maxDim * (0.02 + this.seededRandom(seed + 3) * 0.035);
            const selfRot = rot * (1.5 + this.seededRandom(seed + 4)) * (this.seededRandom(seed + 5) > 0.5 ? 1 : -1);
            const floatHue = (bgHue + this.seededRandom(seed + 6) * 60) % 360;

            const fx = Math.cos(baseAngle) * dist;
            const fy = Math.sin(baseAngle) * dist;

            ctx.save();
            ctx.translate(fx, fy);
            ctx.rotate(selfRot);

            ctx.beginPath();
            for (let s = 0; s <= sides; s++) {
                const a = (Math.PI * 2 * s) / sides;
                const px = Math.cos(a) * size;
                const py = Math.sin(a) * size;
                s === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
            }
            ctx.strokeStyle = `hsla(${floatHue}, ${config.saturation * 0.45}%, 55%, ${baseAlpha * 0.8})`;
            ctx.lineWidth = 0.6 + energy * reactivity;
            ctx.stroke();

            ctx.restore();
        }
        ctx.restore();
    }

    /**
     * Glass background — "Prismatic Refraction Field"
     * 3 parallax layers: prismatic wedges (far), triangular facets (mid), diamond sparkles (near)
     */
    renderGlassBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const baseAlpha = 0.06 + reactivity * 0.10 + energy * reactivity * 0.08;

        // --- FAR LAYER (0.25x): Prismatic light-cone wedges ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.25);

        const wedgeCount = 12;
        for (let i = 0; i < wedgeCount; i++) {
            const angle = (Math.PI * 2 * i) / wedgeCount;
            const wedgeWidth = Math.PI / wedgeCount * 0.6;
            const wedgeHue = (bgHue + i * 30 + harmonic * 20) % 360;
            const reach = maxDim * 1.3;

            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(
                Math.cos(angle - wedgeWidth / 2) * reach,
                Math.sin(angle - wedgeWidth / 2) * reach
            );
            ctx.lineTo(
                Math.cos(angle + wedgeWidth / 2) * reach,
                Math.sin(angle + wedgeWidth / 2) * reach
            );
            ctx.closePath();

            const grad = ctx.createLinearGradient(0, 0,
                Math.cos(angle) * reach * 0.7,
                Math.sin(angle) * reach * 0.7
            );
            grad.addColorStop(0, `hsla(${wedgeHue}, ${config.saturation * 0.6}%, 60%, 0)`);
            grad.addColorStop(0.3, `hsla(${wedgeHue}, ${config.saturation * 0.6}%, 60%, ${baseAlpha * 0.5})`);
            grad.addColorStop(0.7, `hsla(${(wedgeHue + 40) % 360}, ${config.saturation * 0.5}%, 55%, ${baseAlpha * 0.4})`);
            grad.addColorStop(1, `hsla(${(wedgeHue + 80) % 360}, ${config.saturation * 0.4}%, 50%, 0)`);

            ctx.fillStyle = grad;
            ctx.fill();
        }
        ctx.restore();

        // --- MID LAYER (0.6x, counter-rotate): Scattered triangular facets ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.6);

        const facetCount = 20;
        for (let i = 0; i < facetCount; i++) {
            const seed = i * 11 + 137;
            const dist = maxDim * (0.2 + this.seededRandom(seed) * 0.9);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const size = maxDim * (0.03 + this.seededRandom(seed + 2) * 0.05);
            const facetRot = this.seededRandom(seed + 3) * Math.PI + rot * 0.3;
            const facetHue = (bgHue + this.seededRandom(seed + 4) * 80 - 20) % 360;

            const fx = Math.cos(angle) * dist;
            const fy = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(fx, fy);
            ctx.rotate(facetRot);

            // Triangular facet outline
            ctx.beginPath();
            ctx.moveTo(0, -size);
            ctx.lineTo(size * 0.866, size * 0.5);
            ctx.lineTo(-size * 0.866, size * 0.5);
            ctx.closePath();
            ctx.strokeStyle = `hsla(${facetHue}, ${config.saturation * 0.5}%, 60%, ${baseAlpha * 0.8})`;
            ctx.lineWidth = 0.6 + energy * reactivity;
            ctx.stroke();

            // Internal lines from center to vertices
            ctx.beginPath();
            ctx.moveTo(0, 0); ctx.lineTo(0, -size);
            ctx.moveTo(0, 0); ctx.lineTo(size * 0.866, size * 0.5);
            ctx.moveTo(0, 0); ctx.lineTo(-size * 0.866, size * 0.5);
            ctx.strokeStyle = `hsla(${facetHue}, ${config.saturation * 0.4}%, 55%, ${baseAlpha * 0.5})`;
            ctx.lineWidth = 0.4;
            ctx.stroke();

            ctx.restore();
        }
        ctx.restore();

        // --- NEAR LAYER (1.1x): Twinkling diamond sparkle points (4-pointed stars) ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 1.1);

        const sparkleCount = 30;
        for (let i = 0; i < sparkleCount; i++) {
            const seed = i * 13 + 257;
            const dist = maxDim * (0.15 + this.seededRandom(seed) * 1.0);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const phase = this.seededRandom(seed + 2) * Math.PI * 2;
            const baseSize = maxDim * (0.008 + this.seededRandom(seed + 3) * 0.015);
            const size = baseSize * (0.5 + Math.sin(rot * 4 + phase) * 0.5);
            const sparkleHue = (bgHue + this.seededRandom(seed + 4) * 90) % 360;

            const sx = Math.cos(angle) * dist;
            const sy = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(sx, sy);

            // 4-pointed star
            ctx.beginPath();
            ctx.moveTo(0, -size * 2);
            ctx.lineTo(size * 0.3, -size * 0.3);
            ctx.lineTo(size * 2, 0);
            ctx.lineTo(size * 0.3, size * 0.3);
            ctx.lineTo(0, size * 2);
            ctx.lineTo(-size * 0.3, size * 0.3);
            ctx.lineTo(-size * 2, 0);
            ctx.lineTo(-size * 0.3, -size * 0.3);
            ctx.closePath();

            ctx.fillStyle = `hsla(${sparkleHue}, ${config.saturation * 0.6}%, 70%, ${baseAlpha * (0.6 + Math.sin(rot * 4 + phase) * 0.4)})`;
            ctx.fill();

            ctx.restore();
        }
        ctx.restore();
    }

    /**
     * Flower background — "Drifting Pollen Field"
     * 3 parallax layers: sweeping petal curves (far), floating petals (mid), pollen dots with tendrils (near)
     */
    renderFlowerBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const baseAlpha = 0.06 + reactivity * 0.10 + energy * reactivity * 0.08;

        // --- FAR LAYER (0.2x): Large sweeping bezier curves — enormous petal silhouettes ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.2);

        const petalSweeps = 8;
        for (let i = 0; i < petalSweeps; i++) {
            const angle = (Math.PI * 2 * i) / petalSweeps;
            const petalHue = (bgHue + i * 20 + harmonic * 15) % 360;
            const reach = maxDim * (1.0 + energy * reactivity * 0.2);
            const spread = maxDim * 0.4;

            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.bezierCurveTo(
                Math.cos(angle - 0.3) * spread, Math.sin(angle - 0.3) * spread,
                Math.cos(angle + 0.15) * reach * 0.8, Math.sin(angle + 0.15) * reach * 0.8,
                Math.cos(angle) * reach, Math.sin(angle) * reach
            );
            ctx.bezierCurveTo(
                Math.cos(angle - 0.15) * reach * 0.8, Math.sin(angle - 0.15) * reach * 0.8,
                Math.cos(angle + 0.3) * spread, Math.sin(angle + 0.3) * spread,
                0, 0
            );

            ctx.strokeStyle = `hsla(${petalHue}, ${config.saturation * 0.45}%, 55%, ${baseAlpha * 0.7})`;
            ctx.lineWidth = 1.0 + energy * reactivity * 1.5;
            ctx.stroke();
        }
        ctx.restore();

        // --- MID LAYER (0.5x): Floating small petal outlines with center veins ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.5);

        const floatPetals = 16;
        for (let i = 0; i < floatPetals; i++) {
            const seed = i * 9 + 83;
            const dist = maxDim * (0.25 + this.seededRandom(seed) * 0.75);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const petalSize = maxDim * (0.03 + this.seededRandom(seed + 2) * 0.04);
            const petalAngle = this.seededRandom(seed + 3) * Math.PI * 2 + rot * 0.4;
            const petalHue = (bgHue + this.seededRandom(seed + 4) * 50 - 10) % 360;

            const px = Math.cos(angle) * dist;
            const py = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(px, py);
            ctx.rotate(petalAngle);

            // Petal shape via bezier
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.bezierCurveTo(
                petalSize * 0.5, -petalSize * 0.3,
                petalSize * 0.8, -petalSize * 0.15,
                petalSize, 0
            );
            ctx.bezierCurveTo(
                petalSize * 0.8, petalSize * 0.15,
                petalSize * 0.5, petalSize * 0.3,
                0, 0
            );
            ctx.strokeStyle = `hsla(${petalHue}, ${config.saturation * 0.5}%, 55%, ${baseAlpha * 0.8})`;
            ctx.lineWidth = 0.6 + energy * reactivity;
            ctx.stroke();

            // Center vein
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(petalSize * 0.9, 0);
            ctx.strokeStyle = `hsla(${petalHue}, ${config.saturation * 0.4}%, 50%, ${baseAlpha * 0.5})`;
            ctx.lineWidth = 0.4;
            ctx.stroke();

            ctx.restore();
        }
        ctx.restore();

        // --- NEAR LAYER (0.9x, counter-rotate): Pollen dots with vine tendrils ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.9);

        const pollenCount = 25;
        for (let i = 0; i < pollenCount; i++) {
            const seed = i * 17 + 191;
            const dist = maxDim * (0.15 + this.seededRandom(seed) * 0.95);
            const baseAngle = this.seededRandom(seed + 1) * Math.PI * 2;
            // Drift along curved path
            const drift = Math.sin(rot * 2 + this.seededRandom(seed + 2) * Math.PI * 2) * maxDim * 0.03;
            const px = Math.cos(baseAngle) * (dist + drift);
            const py = Math.sin(baseAngle) * (dist + drift);
            const dotSize = maxDim * (0.004 + this.seededRandom(seed + 3) * 0.006);
            const pollenHue = (bgHue + 30 + this.seededRandom(seed + 4) * 40) % 360;

            // Pollen dot
            ctx.beginPath();
            ctx.arc(px, py, dotSize * (0.8 + energy * reactivity * 0.4), 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${pollenHue}, ${config.saturation * 0.5}%, 60%, ${baseAlpha})`;
            ctx.fill();

            // Vine tendril to next pollen (connect some pairs)
            if (i < pollenCount - 1 && this.seededRandom(seed + 5) > 0.5) {
                const nextSeed = (i + 1) * 17 + 191;
                const nextDist = maxDim * (0.15 + this.seededRandom(nextSeed) * 0.95);
                const nextAngle = this.seededRandom(nextSeed + 1) * Math.PI * 2;
                const nextDrift = Math.sin(rot * 2 + this.seededRandom(nextSeed + 2) * Math.PI * 2) * maxDim * 0.03;
                const nx = Math.cos(nextAngle) * (nextDist + nextDrift);
                const ny = Math.sin(nextAngle) * (nextDist + nextDrift);

                const cpx = (px + nx) / 2 + (py - ny) * 0.3;
                const cpy = (py + ny) / 2 + (nx - px) * 0.3;

                ctx.beginPath();
                ctx.moveTo(px, py);
                ctx.quadraticCurveTo(cpx, cpy, nx, ny);
                ctx.strokeStyle = `hsla(${(pollenHue + 20) % 360}, ${config.saturation * 0.35}%, 45%, ${baseAlpha * 0.5})`;
                ctx.lineWidth = 0.4 + energy * reactivity * 0.3;
                ctx.stroke();
            }
        }
        ctx.restore();
    }

    /**
     * Spiral background — "Vortex Current"
     * 3 parallax layers: logarithmic spiral arms (far), wobbling rings (mid), mini comet spirals (near)
     */
    renderSpiralBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const baseAlpha = 0.06 + reactivity * 0.10 + energy * reactivity * 0.08;

        // --- FAR LAYER (0.35x): Large logarithmic spiral arms ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.35);

        const armCount = 4;
        const spiralTurns = 2.5;
        const stepsPerArm = 80;
        for (let arm = 0; arm < armCount; arm++) {
            const armOffset = (Math.PI * 2 * arm) / armCount;
            const armHue = (bgHue + arm * 40 + harmonic * 15) % 360;

            ctx.beginPath();
            for (let s = 0; s <= stepsPerArm; s++) {
                const t = s / stepsPerArm;
                const theta = t * spiralTurns * Math.PI * 2 + armOffset;
                const r = maxDim * 0.05 * Math.pow(1.15, s * 0.15) * (0.9 + energy * reactivity * 0.15);
                const x = Math.cos(theta) * r;
                const y = Math.sin(theta) * r;
                s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.strokeStyle = `hsla(${armHue}, ${config.saturation * 0.5}%, 55%, ${baseAlpha * 0.8})`;
            ctx.lineWidth = 1.0 + energy * reactivity * 1.5;
            ctx.stroke();

            // Traveling pulse along arm
            const pulsePos = ((rot * 8) % 1.0);
            const pulseStep = Math.floor(pulsePos * stepsPerArm);
            const pulseTheta = (pulseStep / stepsPerArm) * spiralTurns * Math.PI * 2 + armOffset;
            const pulseR = maxDim * 0.05 * Math.pow(1.15, pulseStep * 0.15);
            const pulseX = Math.cos(pulseTheta) * pulseR;
            const pulseY = Math.sin(pulseTheta) * pulseR;

            ctx.beginPath();
            ctx.arc(pulseX, pulseY, 2 + energy * 3, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${armHue}, ${config.saturation * 0.6}%, 65%, ${baseAlpha * 1.5})`;
            ctx.fill();
        }
        ctx.restore();

        // --- MID LAYER (0.8x, counter-rotate): Concentric wobbling vortex rings ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.8);

        const vortexRings = 6;
        const ringSteps = 60;
        for (let r = 0; r < vortexRings; r++) {
            const ringRadius = maxDim * (0.15 + r * 0.18);
            const ringHue = (bgHue + r * 30 + brightness * 10) % 360;
            const wobbleAmp = maxDim * 0.015 * (1 + energy * reactivity * 0.5);

            ctx.beginPath();
            for (let s = 0; s <= ringSteps; s++) {
                const angle = (Math.PI * 2 * s) / ringSteps;
                const wobble = Math.sin(angle * (3 + r) + rot * 6) * wobbleAmp;
                const x = Math.cos(angle) * (ringRadius + wobble);
                const y = Math.sin(angle) * (ringRadius + wobble);
                s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.strokeStyle = `hsla(${ringHue}, ${config.saturation * 0.45}%, 55%, ${baseAlpha * (0.9 - r * 0.08)})`;
            ctx.lineWidth = 0.6 + energy * reactivity;
            ctx.stroke();
        }
        ctx.restore();

        // --- NEAR LAYER (1.4x): Mini trailing comet spirals ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 1.4);

        const cometCount = 12;
        for (let i = 0; i < cometCount; i++) {
            const seed = i * 19 + 53;
            const dist = maxDim * (0.2 + this.seededRandom(seed) * 0.8);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const cometSize = maxDim * (0.02 + this.seededRandom(seed + 2) * 0.03);
            const selfRot = rot * (3 + this.seededRandom(seed + 3) * 2) * (this.seededRandom(seed + 4) > 0.5 ? 1 : -1);
            const cometHue = (bgHue + this.seededRandom(seed + 5) * 60) % 360;

            const cx = Math.cos(angle) * dist;
            const cy = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(cx, cy);

            // Mini spiral trail
            const tailSteps = 20;
            ctx.beginPath();
            for (let s = 0; s <= tailSteps; s++) {
                const t = s / tailSteps;
                const theta = t * Math.PI * 3 + selfRot;
                const r = cometSize * t;
                const x = Math.cos(theta) * r;
                const y = Math.sin(theta) * r;
                s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.strokeStyle = `hsla(${cometHue}, ${config.saturation * 0.5}%, 60%, ${baseAlpha * 0.8})`;
            ctx.lineWidth = 0.5 + energy * reactivity * 0.8;
            ctx.stroke();

            // Comet head
            ctx.beginPath();
            const headTheta = Math.PI * 3 + selfRot;
            ctx.arc(Math.cos(headTheta) * cometSize, Math.sin(headTheta) * cometSize,
                1.5 + energy * 2, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${cometHue}, ${config.saturation * 0.6}%, 65%, ${baseAlpha * 1.2})`;
            ctx.fill();

            ctx.restore();
        }
        ctx.restore();
    }

    /**
     * Circuit background — "Grid Matrix"
     * 3 parallax layers: hexagonal grid (far), energy pulses (mid), data nodes (near)
     */
    renderCircuitBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const baseAlpha = 0.06 + reactivity * 0.10 + energy * reactivity * 0.08;

        // --- FAR LAYER (0.15x): Hexagonal grid outlines ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.15);

        const hexSize = maxDim * 0.08;
        const hexH = hexSize * Math.sqrt(3);
        const cols = Math.ceil(maxDim * 1.5 / (hexSize * 1.5));
        const rows = Math.ceil(maxDim * 1.5 / hexH);
        let hexCount = 0;
        const maxHexes = 90;

        for (let row = -rows; row <= rows && hexCount < maxHexes; row++) {
            for (let col = -cols; col <= cols && hexCount < maxHexes; col++) {
                const cx = col * hexSize * 1.5;
                const cy = row * hexH + (col % 2 === 0 ? 0 : hexH * 0.5);

                // Skip hexes outside viewport with margin
                if (Math.abs(cx) > maxDim * 1.2 || Math.abs(cy) > maxDim * 1.2) continue;

                const distFromCenter = Math.sqrt(cx * cx + cy * cy);
                const hexHue = (bgHue + distFromCenter * 0.05 + brightness * 10) % 360;

                ctx.beginPath();
                for (let s = 0; s < 6; s++) {
                    const a = (Math.PI / 3) * s;
                    const hx = cx + Math.cos(a) * hexSize * 0.45;
                    const hy = cy + Math.sin(a) * hexSize * 0.45;
                    s === 0 ? ctx.moveTo(hx, hy) : ctx.lineTo(hx, hy);
                }
                ctx.closePath();
                ctx.strokeStyle = `hsla(${hexHue}, ${config.saturation * 0.4}%, 50%, ${baseAlpha * 0.6})`;
                ctx.lineWidth = 0.5 + energy * reactivity * 0.5;
                ctx.stroke();

                hexCount++;
            }
        }
        ctx.restore();

        // --- MID LAYER (0.4x): Traveling energy pulse dots along trace paths ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.4);

        const traceCount = 15;
        for (let i = 0; i < traceCount; i++) {
            const seed = i * 23 + 97;
            const startAngle = this.seededRandom(seed) * Math.PI * 2;
            const startDist = maxDim * (0.1 + this.seededRandom(seed + 1) * 0.3);
            const endAngle = startAngle + (this.seededRandom(seed + 2) - 0.5) * Math.PI;
            const endDist = maxDim * (0.4 + this.seededRandom(seed + 3) * 0.6);
            const traceHue = (bgHue + this.seededRandom(seed + 4) * 40) % 360;

            const sx = Math.cos(startAngle) * startDist;
            const sy = Math.sin(startAngle) * startDist;
            const ex = Math.cos(endAngle) * endDist;
            const ey = Math.sin(endAngle) * endDist;

            // Trace path (orthogonal segments like circuit traces)
            const mx = ex;
            const my = sy;
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.lineTo(mx, my);
            ctx.lineTo(ex, ey);
            ctx.strokeStyle = `hsla(${traceHue}, ${config.saturation * 0.4}%, 50%, ${baseAlpha * 0.5})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();

            // Traveling pulse dot
            const phase = this.seededRandom(seed + 5);
            const t = ((rot * 3 + phase) % 1.0);
            let px, py;
            if (t < 0.5) {
                const lt = t * 2;
                px = sx + (mx - sx) * lt;
                py = sy + (my - sy) * lt;
            } else {
                const lt = (t - 0.5) * 2;
                px = mx + (ex - mx) * lt;
                py = my + (ey - my) * lt;
            }

            ctx.beginPath();
            ctx.arc(px, py, 1.5 + energy * 2, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${traceHue}, ${config.saturation * 0.6}%, 65%, ${baseAlpha * 1.5})`;
            ctx.fill();
        }
        ctx.restore();

        // --- NEAR LAYER (0.8x, counter-rotate): Glowing data nodes ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.8);

        const nodeCount = 20;
        for (let i = 0; i < nodeCount; i++) {
            const seed = i * 31 + 211;
            const dist = maxDim * (0.15 + this.seededRandom(seed) * 0.9);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const phase = this.seededRandom(seed + 2) * Math.PI * 2;
            const nodeHue = (bgHue + this.seededRandom(seed + 3) * 50) % 360;
            const pulseSize = maxDim * 0.008 * (0.6 + Math.sin(rot * 5 + phase) * 0.4);

            const nx = Math.cos(angle) * dist;
            const ny = Math.sin(angle) * dist;

            // Outer glow
            ctx.beginPath();
            ctx.arc(nx, ny, pulseSize * 3, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${nodeHue}, ${config.saturation * 0.5}%, 55%, ${baseAlpha * 0.3})`;
            ctx.fill();

            // Inner node
            ctx.beginPath();
            ctx.arc(nx, ny, pulseSize, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${nodeHue}, ${config.saturation * 0.6}%, 65%, ${baseAlpha * 1.2})`;
            ctx.fill();

            // Small cross-hair
            const chSize = pulseSize * 2.5;
            ctx.beginPath();
            ctx.moveTo(nx - chSize, ny);
            ctx.lineTo(nx + chSize, ny);
            ctx.moveTo(nx, ny - chSize);
            ctx.lineTo(nx, ny + chSize);
            ctx.strokeStyle = `hsla(${nodeHue}, ${config.saturation * 0.5}%, 60%, ${baseAlpha * 0.6})`;
            ctx.lineWidth = 0.4;
            ctx.stroke();
        }
        ctx.restore();
    }

    /**
     * Fibonacci background — "Sacred Geometry Web"
     * 3 parallax layers: golden spirals (far), golden rectangles (mid), phyllotaxis dots (near)
     */
    renderFibonacciBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const baseAlpha = 0.06 + reactivity * 0.10 + energy * reactivity * 0.08;
        const PHI = (1 + Math.sqrt(5)) / 2;
        const GOLDEN_ANGLE = Math.PI * 2 * (1 - 1 / PHI); // ~137.5 degrees

        // --- FAR LAYER (0.2x): Two large golden spirals (CW + CCW) ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.2);

        for (let dir = 0; dir < 2; dir++) {
            const sign = dir === 0 ? 1 : -1;
            const spiralHue = (bgHue + dir * 30 + harmonic * 15) % 360;
            const steps = 100;

            ctx.beginPath();
            for (let s = 0; s <= steps; s++) {
                const t = s / steps;
                const theta = t * Math.PI * 6 * sign;
                const r = maxDim * 0.02 * Math.pow(PHI, t * 4) * (0.9 + energy * reactivity * 0.15);
                const x = Math.cos(theta) * r;
                const y = Math.sin(theta) * r;
                s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.strokeStyle = `hsla(${spiralHue}, ${config.saturation * 0.5}%, 55%, ${baseAlpha * 0.8})`;
            ctx.lineWidth = 0.8 + energy * reactivity * 1.5;
            ctx.stroke();
        }
        ctx.restore();

        // --- MID LAYER (0.5x, counter-rotate): Nested golden rectangles with quarter-arc spirals ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.5);

        const rectLevels = 8;
        let rw = maxDim * 0.9;
        let rh = rw / PHI;
        let rx = -rw / 2;
        let ry = -rh / 2;

        for (let i = 0; i < rectLevels; i++) {
            const rectHue = (bgHue + i * 20 + brightness * 10) % 360;

            // Rectangle outline
            ctx.beginPath();
            ctx.rect(rx, ry, rw, rh);
            ctx.strokeStyle = `hsla(${rectHue}, ${config.saturation * 0.4}%, 55%, ${baseAlpha * (0.8 - i * 0.06)})`;
            ctx.lineWidth = 0.6 + energy * reactivity * 0.8;
            ctx.stroke();

            // Quarter-circle arc in the square portion
            const squareSize = Math.min(rw, rh);
            let arcCX, arcCY, arcStart;

            switch (i % 4) {
                case 0: // top-left square, arc from bottom-right corner
                    arcCX = rx + squareSize;
                    arcCY = ry + squareSize;
                    arcStart = Math.PI;
                    break;
                case 1: // top-right, arc from bottom-left
                    arcCX = rx + rw - squareSize;
                    arcCY = ry + rh;
                    arcStart = -Math.PI / 2;
                    break;
                case 2: // bottom-right, arc from top-left
                    arcCX = rx + rw - squareSize;
                    arcCY = ry + rh - squareSize;
                    arcStart = 0;
                    break;
                case 3: // bottom-left, arc from top-right
                    arcCX = rx + squareSize;
                    arcCY = ry;
                    arcStart = Math.PI / 2;
                    break;
            }

            ctx.beginPath();
            ctx.arc(arcCX, arcCY, squareSize, arcStart, arcStart + Math.PI / 2);
            ctx.strokeStyle = `hsla(${(rectHue + 40) % 360}, ${config.saturation * 0.5}%, 60%, ${baseAlpha * (0.9 - i * 0.07)})`;
            ctx.lineWidth = 0.5 + energy * reactivity * 0.6;
            ctx.stroke();

            // Subdivide: remove the square and continue with remainder
            const newW = rw - squareSize;
            const newH = rh;
            switch (i % 4) {
                case 0:
                    rx = rx + squareSize;
                    rw = newW > 0 ? newW : rh;
                    rh = newW > 0 ? rh : rw;
                    break;
                case 1:
                    ry = ry + (rh - rw);
                    rh = rw;
                    rw = newH;
                    break;
                case 2:
                    rw = rw - squareSize;
                    if (rw <= 0) { rw = rh; }
                    break;
                case 3:
                    rh = squareSize;
                    rw = newH;
                    break;
            }
            // Prevent degenerate rectangles
            if (rw < 2 || rh < 2) break;
        }
        ctx.restore();

        // --- NEAR LAYER (1.0x): Phyllotaxis dot field ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 1.0);

        const dotCount = 180;
        const maxPhylloRadius = maxDim * 0.85;
        const pulseWave = rot * 4; // outward pulse

        for (let i = 1; i <= dotCount; i++) {
            const angle = i * GOLDEN_ANGLE;
            const r = maxPhylloRadius * Math.sqrt(i / dotCount) * (0.95 + energy * reactivity * 0.1);
            const x = Math.cos(angle) * r;
            const y = Math.sin(angle) * r;

            const normalizedR = r / maxPhylloRadius;
            const pulse = Math.sin(pulseWave - normalizedR * 8) * 0.5 + 0.5;
            const dotSize = (1 + pulse * 2) * (0.8 + energy * reactivity * 0.4);
            const dotHue = (bgHue + i * 0.5 + harmonic * 20) % 360;

            ctx.beginPath();
            ctx.arc(x, y, dotSize, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${dotHue}, ${config.saturation * 0.5}%, 55%, ${baseAlpha * (0.5 + pulse * 0.5)})`;
            ctx.fill();
        }
        ctx.restore();
    }

    /**
     * DMT background — "Hyperspace Warp Field"
     * 3 parallax layers: non-euclidean grid warp (far), pulsing cycloid rings (mid), neon kite shards (near)
     */
    renderDMTBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const baseAlpha = 0.06 + reactivity * 0.10 + energy * reactivity * 0.08;

        // --- FAR LAYER (0.2x): Non-euclidean curved grid radiating from center ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.2);

        // Radial lines that curve with bass
        const gridRadials = 16;
        for (let i = 0; i < gridRadials; i++) {
            const angle = (Math.PI * 2 * i) / gridRadials;
            const lineHue = (bgHue + i * (360 / gridRadials) + harmonic * 30) % 360;
            const curveAmt = maxDim * 0.15 * energy * reactivity;

            ctx.beginPath();
            ctx.moveTo(0, 0);
            const midR = maxDim * 0.6;
            const endR = maxDim * 1.3;
            const cpx = Math.cos(angle + 0.2 * Math.sin(rot * 2)) * midR;
            const cpy = Math.sin(angle + 0.2 * Math.sin(rot * 2)) * midR;
            ctx.quadraticCurveTo(cpx, cpy,
                Math.cos(angle) * endR, Math.sin(angle) * endR);

            ctx.strokeStyle = `hsla(${lineHue}, ${Math.min(100, config.saturation * 0.7)}%, 55%, ${baseAlpha * 0.7})`;
            ctx.lineWidth = 0.5 + energy * reactivity;
            ctx.stroke();
        }

        // Concentric warped rings
        const warpRings = 5;
        for (let r = 0; r < warpRings; r++) {
            const ringR = maxDim * (0.2 + r * 0.22);
            const ringHue = (bgHue + r * 40 + brightness * 20) % 360;
            const steps = 40;

            ctx.beginPath();
            for (let s = 0; s <= steps; s++) {
                const angle = (Math.PI * 2 * s) / steps;
                const warp = Math.sin(angle * 5 + rot * 3) * maxDim * 0.02 * (1 + energy * reactivity);
                const x = Math.cos(angle) * (ringR + warp);
                const y = Math.sin(angle) * (ringR + warp);
                s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.strokeStyle = `hsla(${ringHue}, ${Math.min(100, config.saturation * 0.6)}%, 55%, ${baseAlpha * (0.6 - r * 0.07)})`;
            ctx.lineWidth = 0.6 + energy * reactivity * 0.8;
            ctx.stroke();
        }
        ctx.restore();

        // --- MID LAYER (0.6x, counter-rotate): Cycloid petal rings ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.6);

        const petalRings = 3;
        for (let ring = 0; ring < petalRings; ring++) {
            const ringDist = maxDim * (0.3 + ring * 0.25);
            const petals = 8 + ring * 4;
            const petalSize = maxDim * (0.06 - ring * 0.01);
            const ringHue = (bgHue + ring * 50 + harmonic * 25) % 360;

            for (let p = 0; p < petals; p++) {
                const pAngle = (Math.PI * 2 * p) / petals;
                const px = Math.cos(pAngle) * ringDist;
                const py = Math.sin(pAngle) * ringDist;

                ctx.beginPath();
                ctx.moveTo(px, py);
                const cpDist = petalSize * (0.8 + energy * reactivity * 0.5);
                ctx.bezierCurveTo(
                    px + Math.cos(pAngle - 0.5) * cpDist, py + Math.sin(pAngle - 0.5) * cpDist,
                    px + Math.cos(pAngle + 0.3) * cpDist * 1.5, py + Math.sin(pAngle + 0.3) * cpDist * 1.5,
                    px + Math.cos(pAngle) * petalSize * 2, py + Math.sin(pAngle) * petalSize * 2
                );

                ctx.strokeStyle = `hsla(${(ringHue + p * 15) % 360}, ${Math.min(100, config.saturation * 0.6)}%, 60%, ${baseAlpha * 0.7})`;
                ctx.lineWidth = 0.5 + energy * reactivity * 0.6;
                ctx.stroke();
            }
        }
        ctx.restore();

        // --- NEAR LAYER (1.3x): Floating neon kite/dart shards ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 1.3);

        const shardCount = 20;
        for (let i = 0; i < shardCount; i++) {
            const seed = i * 23 + 333;
            const dist = maxDim * (0.2 + this.seededRandom(seed) * 0.9);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const size = maxDim * (0.015 + this.seededRandom(seed + 2) * 0.025);
            const selfRot = rot * (2 + this.seededRandom(seed + 3) * 2) * (this.seededRandom(seed + 4) > 0.5 ? 1 : -1);
            const shardHue = (bgHue + this.seededRandom(seed + 5) * 120) % 360;

            const sx = Math.cos(angle) * dist;
            const sy = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(sx, sy);
            ctx.rotate(selfRot);

            // Kite shape
            ctx.beginPath();
            ctx.moveTo(0, -size * 1.618);
            ctx.lineTo(size * 0.5, 0);
            ctx.lineTo(0, size * 0.4);
            ctx.lineTo(-size * 0.5, 0);
            ctx.closePath();

            ctx.strokeStyle = `hsla(${shardHue}, ${Math.min(100, config.saturation * 0.7)}%, 60%, ${baseAlpha * 0.9})`;
            ctx.lineWidth = 0.5 + energy * reactivity * 0.5;
            ctx.stroke();

            ctx.restore();
        }
        ctx.restore();
    }

    /**
     * Sacred background — "Cloud Chamber Vapor"
     * 3 parallax layers: radial decay tracks (far), concentric detector rings (mid), Compton scatter bursts (near)
     */
    renderSacredBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const baseAlpha = 0.06 + reactivity * 0.10 + energy * reactivity * 0.08;
        const amberHue = bgHue;
        const crimsonHue = (bgHue - 45 + 360) % 360;

        // --- FAR LAYER (0.15x): Scattered large faint sigils ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.15);

        for (let i = 0; i < 14; i++) {
            const seed = i * 31 + 53;
            const dist = maxDim * (0.2 + this.seededRandom(seed) * 0.85);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const symType = Math.floor(this.seededRandom(seed + 2) * 16);
            const size = maxDim * (0.04 + this.seededRandom(seed + 3) * 0.04);
            const symRot = this.seededRandom(seed + 4) * Math.PI * 2 + rot * 0.2;
            const symHue = (amberHue + this.seededRandom(seed + 5) * 40 - 20) % 360;
            const sx = Math.cos(angle) * dist;
            const sy = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(sx, sy);
            ctx.rotate(symRot);
            ctx.strokeStyle = `hsla(${symHue}, ${config.saturation * 0.4}%, 50%, ${baseAlpha * 0.6})`;
            ctx.lineWidth = 0.6 + energy * reactivity * 0.3;
            ctx.lineCap = 'round';
            this._drawOccultSymbol(ctx, symType, size);
            ctx.restore();
        }
        ctx.restore();

        // --- MID LAYER (-0.5x, counter-rotate): Dense medium symbols ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.5);

        for (let i = 0; i < 20; i++) {
            const seed = i * 19 + 137;
            const dist = maxDim * (0.1 + this.seededRandom(seed) * 0.95);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const symType = Math.floor(this.seededRandom(seed + 2) * 16);
            const size = maxDim * (0.015 + this.seededRandom(seed + 3) * 0.025);
            const symRot = this.seededRandom(seed + 4) * Math.PI * 2 + rot * 0.4;
            const symHue = (crimsonHue + this.seededRandom(seed + 5) * 50 - 25) % 360;
            const sx = Math.cos(angle) * dist;
            const sy = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(sx, sy);
            ctx.rotate(symRot);
            ctx.strokeStyle = `hsla(${symHue}, ${config.saturation * 0.5}%, 45%, ${baseAlpha * 0.5})`;
            ctx.lineWidth = 0.5 + harmonic * 0.4;
            ctx.lineCap = 'round';
            this._drawOccultSymbol(ctx, symType, size);
            ctx.restore();
        }
        ctx.restore();

        // --- NEAR LAYER (0.9x): Small bright glyphs ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.9);

        for (let i = 0; i < 18; i++) {
            const seed = i * 23 + 211;
            const dist = maxDim * (0.08 + this.seededRandom(seed) * 1.0);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const symType = Math.floor(this.seededRandom(seed + 2) * 16);
            const size = maxDim * (0.008 + this.seededRandom(seed + 3) * 0.014);
            const phase = this.seededRandom(seed + 4) * Math.PI * 2;
            const pulse = 0.6 + Math.sin(rot * 3 + phase) * 0.4;
            const symHue = (bgHue + this.seededRandom(seed + 5) * 60) % 360;
            const sx = Math.cos(angle) * dist;
            const sy = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(sx, sy);
            ctx.rotate(rot * 0.6 + phase);
            ctx.strokeStyle = `hsla(${symHue}, ${config.saturation * 0.5}%, 60%, ${baseAlpha * 0.8 * pulse})`;
            ctx.lineWidth = 0.4 + brightness * 0.4;
            ctx.lineCap = 'round';
            this._drawOccultSymbol(ctx, symType, size);
            ctx.restore();
        }
        ctx.restore();
    }

    /**
     * Mycelial background — "Mushroom Forest Floor"
     * 3 parallax layers: ghostly mushroom silhouettes (far), drifting spore haze (mid), glowing fairy-ring fungi (near)
     */
    renderMycelialBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const baseAlpha = 0.06 + reactivity * 0.10 + energy * reactivity * 0.08;

        // --- FAR LAYER (0.15x): Ghostly Vein Network ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.15);

        for (let i = 0; i < 12; i++) {
            const seed = i * 37 + 47;
            const startDist = maxDim * (0.05 + this.seededRandom(seed) * 0.3);
            const startAngle = this.seededRandom(seed + 1) * Math.PI * 2;
            const endDist = maxDim * (0.4 + this.seededRandom(seed + 2) * 0.5);
            const endAngle = startAngle + (this.seededRandom(seed + 3) - 0.5) * 1.5;
            const veinHue = (bgHue + this.seededRandom(seed + 4) * 40 - 20) % 360;

            const x1 = Math.cos(startAngle) * startDist;
            const y1 = Math.sin(startAngle) * startDist;
            const x2 = Math.cos(endAngle) * endDist;
            const y2 = Math.sin(endAngle) * endDist;
            // Wobble control point gently with rotation
            const wobble = Math.sin(rot * 0.5 + this.seededRandom(seed + 7) * Math.PI * 2) * maxDim * 0.02;
            const cpx = (x1 + x2) / 2 + (this.seededRandom(seed + 5) - 0.5) * maxDim * 0.3 + wobble;
            const cpy = (y1 + y2) / 2 + (this.seededRandom(seed + 6) - 0.5) * maxDim * 0.3 + wobble;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.quadraticCurveTo(cpx, cpy, x2, y2);
            ctx.strokeStyle = `hsla(${veinHue}, ${config.saturation * 0.3}%, 40%, ${baseAlpha * (0.5 + this.seededRandom(seed + 8) * 0.5)})`;
            ctx.lineWidth = 0.5 + this.seededRandom(seed + 9) * 0.5;
            ctx.lineCap = 'round';
            ctx.stroke();
        }
        ctx.restore();

        // --- MID LAYER (-0.5x): Floating Nutrient Particles ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.5);

        for (let i = 0; i < 18; i++) {
            const seed = i * 13 + 113;
            const orbitR = maxDim * (0.1 + this.seededRandom(seed) * 0.8);
            const baseAngle = this.seededRandom(seed + 1) * Math.PI * 2;
            const speed = 0.3 + this.seededRandom(seed + 2) * 0.4;
            const angle = baseAngle + rot * speed * 0.1;
            const particleHue = (bgHue + 95 + this.seededRandom(seed + 3) * 40) % 360;
            const size = maxDim * (0.002 + this.seededRandom(seed + 4) * 0.003);

            const px = Math.cos(angle) * orbitR;
            const py = Math.sin(angle) * orbitR;
            const alpha = baseAlpha * (0.3 + reactivity * 0.6);

            const pGrad = ctx.createRadialGradient(px, py, 0, px, py, size * 3);
            pGrad.addColorStop(0, `hsla(${particleHue}, ${config.saturation * 0.5}%, 60%, ${alpha})`);
            pGrad.addColorStop(1, `hsla(${particleHue}, ${config.saturation * 0.3}%, 45%, 0)`);
            ctx.beginPath();
            ctx.arc(px, py, size * 3, 0, Math.PI * 2);
            ctx.fillStyle = pGrad;
            ctx.fill();
        }
        ctx.restore();

        // --- NEAR LAYER (1.0x): Bright Growth Nodes ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 1.0);

        for (let i = 0; i < 8; i++) {
            const seed = i * 29 + 227;
            const dist = maxDim * (0.15 + this.seededRandom(seed) * 0.7);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const phase = this.seededRandom(seed + 2) * Math.PI * 2;
            const pulse = 0.5 + Math.sin(rot * 3 + phase) * 0.5;
            const nodeHue = (bgHue + this.seededRandom(seed + 3) * 60) % 360;
            const nx = Math.cos(angle) * dist;
            const ny = Math.sin(angle) * dist;
            const nodeSize = maxDim * (0.003 + this.seededRandom(seed + 4) * 0.005) * (0.7 + pulse * 0.5);
            const alpha = baseAlpha * (0.6 + reactivity * 1.0) * pulse;

            const nGrad = ctx.createRadialGradient(nx, ny, 0, nx, ny, nodeSize * 3);
            nGrad.addColorStop(0, `hsla(${nodeHue}, ${config.saturation * 0.6}%, 65%, ${alpha})`);
            nGrad.addColorStop(1, `hsla(${nodeHue}, ${config.saturation * 0.4}%, 45%, 0)`);
            ctx.beginPath();
            ctx.arc(nx, ny, nodeSize * 3, 0, Math.PI * 2);
            ctx.fillStyle = nGrad;
            ctx.fill();
        }
        ctx.restore();
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

    renderFrame(frameData, deltaTime, isPlaying) {
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
        const canvasRadius = Math.min(this.canvas.width, this.canvas.height) * 0.48;
        const sizeNorm = config.baseRadius / 150;
        const scale = 1 + (this.smoothedValues.percussiveImpact * (config.maxScale - 1));
        const radius = canvasRadius * sizeNorm * scale;

        // Rotation accumulation (time-based, not frame-based)
        const rotationDelta = this.smoothedValues.harmonicEnergy * config.rotationSpeed * (deltaTime / 1000);
        if (isPlaying || this.smoothedValues.harmonicEnergy > 0.1) {
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

        // Per-style size normalization — keeps visual footprint consistent across styles
        const sizeScale = {
            circuit: 0.75, fibonacci: 1.15, dmt: 1.1, fluid: 1.5, quark: 1.3, fractal: 1.2
        }[config.style] || 1.0;
        const scaledRadius = radius * sizeScale;

        // Render based on selected style
        switch (config.style) {
            case 'glass':
                this.renderGlassStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'flower':
                this.renderFlowerStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'spiral':
                this.renderSpiralStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'circuit':
                this.renderCircuitStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'fibonacci':
                this.renderFibonacciStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'dmt':
                this.renderDMTStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'sacred':
                this.renderSacredStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'mycelial':
                this.renderMycelialStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'fluid':
                this.renderFluidStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'orrery':
                this.renderOrreryStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'quark':
                this.renderQuarkStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'fractal':
                this.renderFractalStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
            case 'geometric':
            default:
                this.renderGeometricStyle(ctx, centerX, centerY, scaledRadius, numSides, hue, thickness);
                break;
        }
    }

    /**
     * Geometric style - orbiting polygons with radial symmetry (uses shapeSeed for variation)
     * All mirrors are IDENTICAL to maintain kaleidoscope symmetry
     */
    renderGeometricStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const orbitFactor = config.orbitRadius / 200;
        const orbitDistance = radius * orbitFactor * (0.5 + harmonic * 0.5);

        // Seed-based variation parameters (GLOBAL - same for all mirrors)
        const rotationDir = this.seededRandom(seed) > 0.5 ? 1 : -1;
        const innerRotationSpeed = 1 + this.seededRandom(seed + 1) * 1.5;
        const hueShift = this.seededRandom(seed + 2) * 60;
        const sizeMultiplier = 0.8 + this.seededRandom(seed + 3) * 0.4;
        const outerSidesBonus = Math.floor(this.seededRandom(seed + 4) * 3);
        const innerSidesBonus = Math.floor(this.seededRandom(seed + 5) * 2);
        const hasTinyPolygon = this.seededRandom(seed + 6) > 0.4;
        const tinySides = 3 + Math.floor(this.seededRandom(seed + 7) * 3);

        // Draw kaleidoscope pattern - ALL MIRRORS IDENTICAL
        for (let i = 0; i < config.mirrors; i++) {
            const mirrorAngle = (Math.PI * 2 * i / config.mirrors) + this.accumulatedRotation * 0.3 * rotationDir;

            const orbitX = centerX + orbitDistance * Math.cos(mirrorAngle);
            const orbitY = centerY + orbitDistance * Math.sin(mirrorAngle);

            // Outer polygon (same for all mirrors)
            const outerSides = numSides + outerSidesBonus;
            this.drawPolygon(
                ctx,
                orbitX, orbitY,
                radius * 0.8 * sizeMultiplier,
                outerSides,
                this.accumulatedRotation * rotationDir + mirrorAngle,
                `hsl(${hue}, ${config.saturation}%, ${65 + energy * 15}%)`,
                thickness * (0.8 + energy * 0.4)
            );

            // Inner polygon (counter-rotating, same for all mirrors)
            const innerHue = (hue + 180 + hueShift) % 360;
            const innerSides = Math.max(3, outerSides - 1 - innerSidesBonus);
            this.drawPolygon(
                ctx,
                orbitX, orbitY,
                radius * 0.35 * sizeMultiplier,
                innerSides,
                -this.accumulatedRotation * innerRotationSpeed * rotationDir + mirrorAngle,
                `hsl(${innerHue}, ${config.saturation * 0.8}%, ${55 + energy * 20}%)`,
                Math.max(1, thickness / 2)
            );

            // Extra tiny polygon (same for all mirrors if enabled)
            if (hasTinyPolygon) {
                const tinyHue = (hue + 90) % 360;
                this.drawPolygon(
                    ctx,
                    orbitX, orbitY,
                    radius * 0.15 * (1 + energy * 0.5),
                    tinySides,
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
     * 2D value noise — smooth pseudo-random field for organic textures.
     * Uses seededRandom as the hash function with bilinear interpolation.
     */
    _fractalNoise(x, y) {
        const ix = Math.floor(x);
        const iy = Math.floor(y);
        const fx = x - ix;
        const fy = y - iy;
        // Smoothstep
        const ux = fx * fx * (3 - 2 * fx);
        const uy = fy * fy * (3 - 2 * fy);
        // Corner hash values
        const n00 = this.seededRandom(ix * 374761 + iy * 668265);
        const n10 = this.seededRandom((ix + 1) * 374761 + iy * 668265);
        const n01 = this.seededRandom(ix * 374761 + (iy + 1) * 668265);
        const n11 = this.seededRandom((ix + 1) * 374761 + (iy + 1) * 668265);
        // Bilinear interpolation
        return n00 * (1 - ux) * (1 - uy) + n10 * ux * (1 - uy) +
               n01 * (1 - ux) * uy + n11 * ux * uy;
    }

    /**
     * Fractal Brownian Motion — multi-octave noise for rich, layered textures.
     * Sums noise at increasing frequencies and decreasing amplitudes.
     */
    _fractalFbm(x, y, octaves) {
        let value = 0;
        let amplitude = 0.5;
        let frequency = 1;
        for (let i = 0; i < octaves; i++) {
            value += amplitude * this._fractalNoise(x * frequency, y * frequency);
            amplitude *= 0.5;
            frequency *= 2.0;
        }
        return value;
    }

    /**
     * Glass style - faceted kaleidoscope with gem-like internal reflections
     * Inspired by real kaleidoscope optics: triangular tessellation, layered depth,
     * prismatic color shifts, and nested mirror patterns
     */
    renderGlassStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const mirrors = config.mirrors * 2;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const seed = config.shapeSeed;
        const orbitFactor = config.orbitRadius / 200;

        ctx.save();
        ctx.translate(centerX, centerY);

        const globalRotation = this.accumulatedRotation * 0.2;
        ctx.rotate(globalRotation);

        // Draw mirrored wedges with faceted content
        for (let m = 0; m < mirrors; m++) {
            ctx.save();
            ctx.rotate((Math.PI * 2 * m) / mirrors);
            if (m % 2 === 1) {
                ctx.scale(-1, 1);
            }
            this.drawGlassWedge(ctx, radius, hue, thickness, seed, m, numSides, orbitFactor);
            ctx.restore();
        }

        ctx.restore();

        // Multi-layered central jewel
        const jewelSize = radius * (0.3 + energy * 0.25);
        this.drawCentralJewel(ctx, centerX, centerY, jewelSize, hue, energy, numSides, thickness);
    }

    drawGlassWedge(ctx, maxRadius, hue, thickness, baseSeed, wedgeIndex, numSides, orbitFactor) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const wedgeAngle = Math.PI / config.mirrors;

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, maxRadius, 0, wedgeAngle);
        ctx.closePath();
        ctx.clip();

        // --- Layer 0: Triangular tessellation (real kaleidoscope grid) ---
        const tessSize = 40 + config.glassSlices * 1.5 + brightness * 20;
        const triH = tessSize * Math.sqrt(3) / 2;
        const rows = Math.ceil(maxRadius / triH) + 2;
        const cols = Math.ceil(maxRadius / tessSize) + 2;

        for (let row = -1; row < rows; row++) {
            for (let col = -1; col < cols; col++) {
                const cx = col * tessSize + (row % 2 === 0 ? 0 : tessSize * 0.5);
                const cy = row * triH;
                const dist = Math.sqrt(cx * cx + cy * cy);
                if (dist > maxRadius * 1.1) continue;

                // Check if roughly in wedge
                const ptAngle = Math.atan2(cy, cx);
                if (ptAngle < -0.15 || ptAngle > wedgeAngle + 0.15) continue;

                const distRatio = dist / maxRadius;
                const triSeed = baseSeed + row * 137 + col * 59;
                const gemHue = (hue + this.seededRandom(triSeed) * 90 - 30 + harmonic * 40 + distRatio * 60) % 360;
                const gemLightness = 35 + brightness * 20 + energy * 15 + this.seededRandom(triSeed + 1) * 15;
                const gemAlpha = (0.15 + energy * 0.25 + harmonic * 0.1) * (1 - distRatio * 0.4);

                // Upward triangle
                const triScale = tessSize * 0.48 * (0.85 + energy * 0.3 + Math.sin(this.accumulatedRotation * 2 + dist * 0.01) * 0.05);
                ctx.beginPath();
                ctx.moveTo(cx, cy - triScale * 0.6);
                ctx.lineTo(cx + triScale * 0.5, cy + triScale * 0.35);
                ctx.lineTo(cx - triScale * 0.5, cy + triScale * 0.35);
                ctx.closePath();

                ctx.fillStyle = `hsla(${gemHue}, ${config.saturation}%, ${gemLightness}%, ${gemAlpha})`;
                ctx.fill();
                ctx.strokeStyle = `hsla(${gemHue}, ${config.saturation * 0.8}%, ${gemLightness + 20}%, ${gemAlpha * 0.7})`;
                ctx.lineWidth = thickness * (0.15 + energy * 0.25);
                ctx.stroke();

                // Downward triangle (inverted)
                const invHue = (gemHue + 30 + brightness * 20) % 360;
                ctx.beginPath();
                ctx.moveTo(cx + tessSize * 0.5, cy + triScale * 0.35);
                ctx.lineTo(cx, cy + triH * 0.9);
                ctx.lineTo(cx - tessSize * 0.05, cy + triScale * 0.35);
                ctx.closePath();

                ctx.fillStyle = `hsla(${invHue}, ${config.saturation * 0.9}%, ${gemLightness + 8}%, ${gemAlpha * 0.8})`;
                ctx.fill();
            }
        }

        // --- Layer 1: Hexagonal gem facets at multiple depths ---
        const numGems = 4 + Math.floor(config.glassSlices / 10);
        for (let i = 0; i < numGems; i++) {
            const seed = baseSeed + 500 + i * 23.7;
            const dist = maxRadius * (0.15 + this.seededRandom(seed) * 0.65) * orbitFactor;
            const angle = this.seededRandom(seed + 1) * wedgeAngle * 0.9;
            const gemSize = (30 + this.seededRandom(seed + 2) * 60) * (0.8 + energy * 0.5);
            const gemHue = (hue + this.seededRandom(seed + 3) * 80 - 30 + harmonic * 40) % 360;
            const gemRotation = this.seededRandom(seed + 4) * Math.PI + this.accumulatedRotation * (0.3 + this.seededRandom(seed + 5) * 0.4);

            const x = Math.cos(angle) * dist;
            const y = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(gemRotation);

            // Draw gem with internal facet lines (sides from numSides knob)
            const sides = numSides;
            const points = [];
            for (let j = 0; j < sides; j++) {
                const a = (Math.PI * 2 * j) / sides;
                points.push({ x: Math.cos(a) * gemSize * 0.5, y: Math.sin(a) * gemSize * 0.5 });
            }

            // Outer hex
            ctx.beginPath();
            points.forEach((p, idx) => idx === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
            ctx.closePath();

            const alpha = 0.2 + energy * 0.35 + harmonic * 0.1;
            ctx.fillStyle = `hsla(${gemHue}, ${config.saturation}%, ${45 + brightness * 20}%, ${alpha})`;
            ctx.fill();
            ctx.strokeStyle = `hsla(${gemHue}, ${config.saturation}%, ${70 + energy * 15}%, ${alpha + 0.25})`;
            ctx.lineWidth = thickness * (0.3 + energy * 0.4);
            ctx.stroke();

            // Internal facet lines - from center to each vertex (gem cut pattern)
            for (let j = 0; j < sides; j++) {
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(points[j].x, points[j].y);
                ctx.strokeStyle = `hsla(${(gemHue + 20) % 360}, ${config.saturation * 0.7}%, 75%, ${alpha * 0.5})`;
                ctx.lineWidth = thickness * (0.15 + energy * 0.15);
                ctx.stroke();
            }

            // Inner facet (smaller polygon rotated) - creates depth
            const innerScale = 0.5 + energy * 0.15;
            const innerRot = Math.PI / sides;
            ctx.beginPath();
            for (let j = 0; j < sides; j++) {
                const a = (Math.PI * 2 * j) / sides + innerRot;
                const px = Math.cos(a) * gemSize * 0.5 * innerScale;
                const py = Math.sin(a) * gemSize * 0.5 * innerScale;
                j === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
            }
            ctx.closePath();
            ctx.fillStyle = `hsla(${(gemHue + 40) % 360}, ${config.saturation}%, ${60 + energy * 25}%, ${alpha * 0.6})`;
            ctx.fill();
            ctx.strokeStyle = `hsla(${(gemHue + 40) % 360}, ${config.saturation * 0.8}%, 80%, ${alpha * 0.4})`;
            ctx.lineWidth = thickness * 0.15;
            ctx.stroke();

            ctx.restore();
        }

        // --- Layer 2: Prismatic light rays from center ---
        const numRays = 3 + Math.floor(brightness * 4);
        for (let i = 0; i < numRays; i++) {
            const seed = baseSeed + 700 + i * 17;
            const rayAngle = this.seededRandom(seed) * wedgeAngle * 0.85;
            const rayWidth = (0.01 + this.seededRandom(seed + 1) * 0.03) * (1 + energy * 0.5);
            const rayHue = (hue + i * 25 + this.seededRandom(seed + 2) * 30) % 360;

            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(
                Math.cos(rayAngle - rayWidth) * maxRadius,
                Math.sin(rayAngle - rayWidth) * maxRadius
            );
            ctx.lineTo(
                Math.cos(rayAngle + rayWidth) * maxRadius,
                Math.sin(rayAngle + rayWidth) * maxRadius
            );
            ctx.closePath();

            const grad = ctx.createLinearGradient(0, 0,
                Math.cos(rayAngle) * maxRadius, Math.sin(rayAngle) * maxRadius);
            grad.addColorStop(0, `hsla(${rayHue}, ${config.saturation}%, 85%, ${0.15 + energy * 0.2})`);
            grad.addColorStop(0.5, `hsla(${(rayHue + 30) % 360}, ${config.saturation}%, 70%, ${0.08 + energy * 0.12})`);
            grad.addColorStop(1, `hsla(${(rayHue + 60) % 360}, ${config.saturation}%, 55%, 0)`);
            ctx.fillStyle = grad;
            ctx.fill();
        }

        // --- Layer 3: Scattered gem sparkles ---
        const sparkleCount = config.glassSlices + Math.floor(brightness * 15);
        for (let i = 0; i < sparkleCount; i++) {
            const seed = baseSeed + 200 + i * 7.1;
            const dist = maxRadius * (0.1 + this.seededRandom(seed + 1) * 0.85);
            const angle = this.seededRandom(seed + 2) * wedgeAngle;
            const x = Math.cos(angle) * dist;
            const y = Math.sin(angle) * dist;

            const sparkleSize = (3 + this.seededRandom(seed) * 8) * (0.6 + energy * 1.0 + brightness * 0.3);
            const sparkleHue = (hue + this.seededRandom(seed + 3) * 70 + harmonic * 40) % 360;
            const sparkleAlpha = (0.3 + energy * 0.6) * (1 - dist / maxRadius * 0.5);

            // Diamond sparkle shape (4-pointed star)
            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(this.seededRandom(seed + 4) * Math.PI + this.accumulatedRotation * this.seededRandom(seed + 5));

            ctx.beginPath();
            ctx.moveTo(0, -sparkleSize);
            ctx.lineTo(sparkleSize * 0.25, 0);
            ctx.lineTo(0, sparkleSize);
            ctx.lineTo(-sparkleSize * 0.25, 0);
            ctx.closePath();

            ctx.fillStyle = `hsla(${sparkleHue}, ${config.saturation}%, ${65 + brightness * 25}%, ${sparkleAlpha})`;
            ctx.fill();

            ctx.restore();
        }

        // --- Layer 4: Bright crystal highlights (pulse with beats) ---
        for (let i = 0; i < 4; i++) {
            const seed = baseSeed + 300 + i * 31.3;
            const baseSize = 20 + this.seededRandom(seed) * 35;
            const size = baseSize * (0.6 + energy * 0.8);
            const dist = maxRadius * (0.2 + this.seededRandom(seed + 1) * 0.45);
            const angle = this.seededRandom(seed + 2) * wedgeAngle * 0.8;
            const shapeHue = (hue + 20 + this.seededRandom(seed + 3) * 40) % 360;

            const x = Math.cos(angle) * dist;
            const y = Math.sin(angle) * dist;

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(this.seededRandom(seed + 4) * Math.PI + this.accumulatedRotation * 0.6);

            // Elongated crystal shard
            ctx.beginPath();
            ctx.moveTo(0, -size * 1.2);
            ctx.lineTo(size * 0.3, -size * 0.2);
            ctx.lineTo(size * 0.15, size * 0.8);
            ctx.lineTo(-size * 0.15, size * 0.8);
            ctx.lineTo(-size * 0.3, -size * 0.2);
            ctx.closePath();

            const alpha = 0.35 + energy * 0.45;
            // Gradient fill for glass depth
            const crystalGrad = ctx.createLinearGradient(0, -size * 1.2, 0, size * 0.8);
            crystalGrad.addColorStop(0, `hsla(${shapeHue}, ${config.saturation}%, ${80 + energy * 15}%, ${alpha})`);
            crystalGrad.addColorStop(0.4, `hsla(${(shapeHue + 25) % 360}, ${config.saturation * 0.9}%, ${65 + energy * 20}%, ${alpha * 0.7})`);
            crystalGrad.addColorStop(1, `hsla(${(shapeHue + 50) % 360}, ${config.saturation * 0.8}%, ${50 + energy * 10}%, ${alpha * 0.4})`);
            ctx.fillStyle = crystalGrad;
            ctx.fill();

            ctx.strokeStyle = `hsla(${shapeHue}, ${config.saturation}%, 90%, ${alpha * 0.6})`;
            ctx.lineWidth = thickness * (0.25 + energy * 0.15);
            ctx.stroke();

            // Internal facet line
            ctx.beginPath();
            ctx.moveTo(0, -size * 1.2);
            ctx.lineTo(0, size * 0.8);
            ctx.strokeStyle = `hsla(${shapeHue}, ${config.saturation * 0.5}%, 85%, ${alpha * 0.3})`;
            ctx.lineWidth = thickness * 0.15;
            ctx.stroke();

            ctx.restore();
        }

        ctx.restore();
    }

    drawCentralJewel(ctx, centerX, centerY, radius, hue, energy, numSides, thickness) {
        const config = this.config;
        numSides = numSides || 6;
        thickness = thickness || 3;

        // Pulsing outer glow
        const glowSize = radius * (1.8 + energy * 0.6);
        const glowGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, glowSize);
        glowGradient.addColorStop(0, `hsla(${hue}, ${config.saturation}%, ${85 + energy * 10}%, ${0.5 + energy * 0.3})`);
        glowGradient.addColorStop(0.4, `hsla(${(hue + 20) % 360}, ${config.saturation}%, 65%, ${0.2 + energy * 0.15})`);
        glowGradient.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 40%, 0)`);
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, glowSize, 0, Math.PI * 2);
        ctx.fill();

        // Faceted jewel - polygon with internal geometry
        const sides = numSides;
        const jewPoints = [];
        for (let i = 0; i < sides; i++) {
            const a = (Math.PI * 2 * i) / sides + this.accumulatedRotation * 0.3;
            jewPoints.push({
                x: centerX + Math.cos(a) * radius,
                y: centerY + Math.sin(a) * radius
            });
        }

        // Outer faceted shape
        ctx.beginPath();
        jewPoints.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
        ctx.closePath();

        const jewelGrad = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
        jewelGrad.addColorStop(0, `hsla(${hue}, ${config.saturation}%, ${90 + energy * 10}%, ${0.9 + energy * 0.1})`);
        jewelGrad.addColorStop(0.5, `hsla(${(hue + 30) % 360}, ${config.saturation}%, ${65 + energy * 20}%, ${0.65 + energy * 0.2})`);
        jewelGrad.addColorStop(1, `hsla(${(hue + 60) % 360}, ${config.saturation}%, 45%, ${0.3 + energy * 0.15})`);
        ctx.fillStyle = jewelGrad;
        ctx.fill();

        // Facet lines from center to vertices
        for (let i = 0; i < sides; i++) {
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(jewPoints[i].x, jewPoints[i].y);
            ctx.strokeStyle = `hsla(${(hue + i * 20) % 360}, ${config.saturation * 0.6}%, 80%, ${0.25 + energy * 0.2})`;
            ctx.lineWidth = thickness * (0.25 + energy * 0.15);
            ctx.stroke();
        }

        // Inner rotated polygon
        const innerRadius = radius * (0.55 + energy * 0.1);
        ctx.beginPath();
        for (let i = 0; i < sides; i++) {
            const a = (Math.PI * 2 * i) / sides + this.accumulatedRotation * 0.3 + Math.PI / sides;
            const px = centerX + Math.cos(a) * innerRadius;
            const py = centerY + Math.sin(a) * innerRadius;
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.strokeStyle = `hsla(${(hue + 40) % 360}, ${config.saturation}%, 80%, ${0.3 + energy * 0.3})`;
        ctx.lineWidth = thickness * (0.3 + energy * 0.3);
        ctx.stroke();

        // Bright pulsing core
        const coreRadius = radius * (0.3 + energy * 0.12);
        const coreGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, coreRadius);
        coreGradient.addColorStop(0, `hsla(${hue}, ${config.saturation * 0.4}%, ${97 + energy * 3}%, 1)`);
        coreGradient.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 75%, ${0.3 + energy * 0.4})`);
        ctx.fillStyle = coreGradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, coreRadius, 0, Math.PI * 2);
        ctx.fill();
    }

    /**
     * Flower style - petal-like shapes radiating from center (uses shapeSeed for variation)
     * All petals in each layer are IDENTICAL to maintain kaleidoscope symmetry
     */
    renderFlowerStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const mirrors = config.mirrors;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const orbitFactor = config.orbitRadius / 200;

        // Seed-based variation parameters (GLOBAL - same for all petals)
        const rotationDir = this.seededRandom(seed) > 0.5 ? 1 : -1;
        const petalShape = this.seededRandom(seed + 1); // 0-1 affects curve shape
        const hueSpread = 20 + this.seededRandom(seed + 2) * 60;
        const layerCount = Math.max(2, numSides - 1) + Math.floor(this.seededRandom(seed + 3) * 2);
        const hasFill = this.seededRandom(seed + 4) > 0.5;
        const hasVein = this.seededRandom(seed + 5) > 0.4;
        const curveStyle1 = 0.2 + this.seededRandom(seed + 6) * 0.2;
        const curveStyle2 = 0.7 + this.seededRandom(seed + 7) * 0.2;
        const widthMult1 = 1 + this.seededRandom(seed + 8) * 0.3;
        const widthMult2 = 0.4 + this.seededRandom(seed + 9) * 0.3;

        ctx.save();
        ctx.translate(centerX, centerY);

        // Multiple layers of petals - ALL petals in a layer are identical
        for (let layer = 0; layer < layerCount; layer++) {
            const layerSeed = seed + layer * 31;
            const layerRadius = radius * orbitFactor * (0.4 + layer * 0.35) * (1 + energy * 0.3);
            const rotSpeed = (1 - layer * 0.25) * (this.seededRandom(layerSeed) > 0.5 ? 1 : -1);
            const layerRotation = this.accumulatedRotation * rotSpeed * rotationDir;
            const petalCount = mirrors + layer * 2;
            const layerHue = (hue + layer * hueSpread) % 360;

            // Per-layer seed-based parameters (same for all petals in this layer)
            const layerPetalLength = layerRadius * (0.7 + harmonic * 0.5);
            const layerPetalWidth = layerRadius * (0.25 + petalShape * 0.15) * (1 + energy * 0.5);

            ctx.save();
            ctx.rotate(layerRotation);

            for (let i = 0; i < petalCount; i++) {
                const petalAngle = (Math.PI * 2 * i) / petalCount;

                ctx.save();
                ctx.rotate(petalAngle);

                // Draw petal shape - ALL IDENTICAL
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.bezierCurveTo(
                    layerPetalWidth * widthMult1, layerPetalLength * curveStyle1,
                    layerPetalWidth * widthMult2, layerPetalLength * curveStyle2,
                    0, layerPetalLength
                );
                ctx.bezierCurveTo(
                    -layerPetalWidth * widthMult2, layerPetalLength * curveStyle2,
                    -layerPetalWidth * widthMult1, layerPetalLength * curveStyle1,
                    0, 0
                );

                const alpha = 0.35 + brightness * 0.35 + energy * 0.2 - layer * 0.08;
                ctx.strokeStyle = `hsla(${layerHue}, ${config.saturation}%, ${55 + layer * 8 + energy * 15}%, ${alpha})`;
                ctx.lineWidth = thickness * (1 - layer * 0.15) * (0.8 + energy * 0.4);
                ctx.stroke();

                // Optional fill (same for all petals in layer)
                if (hasFill) {
                    ctx.fillStyle = `hsla(${layerHue}, ${config.saturation * 0.8}%, ${60 + energy * 20}%, ${alpha * 0.3})`;
                    ctx.fill();
                }

                // Inner vein (same for all petals in layer)
                if (hasVein) {
                    ctx.beginPath();
                    ctx.moveTo(0, layerPetalLength * 0.15);
                    ctx.lineTo(0, layerPetalLength * 0.85);
                    ctx.strokeStyle = `hsla(${layerHue}, ${config.saturation}%, 75%, ${alpha * 0.4})`;
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

        // Center dots (identical spacing)
        if (this.seededRandom(seed + 22) > 0.4) {
            const dotCount = 5 + Math.floor(this.seededRandom(seed + 23) * 4);
            const dotDist = stamenRadius * 0.6;
            const dotSize = 2 + energy * 3;
            for (let d = 0; d < dotCount; d++) {
                const dotAngle = (Math.PI * 2 * d) / dotCount + this.accumulatedRotation * 0.5;
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
        const orbitFactor = config.orbitRadius / 200;

        // LAYER 0: Deep background spirals - slow, ghostly
        ctx.save();
        ctx.rotate(-this.accumulatedRotation * 0.15);
        this.drawFractalLayer(ctx, arms, radius * orbitFactor, hue, seed, 0, energy, harmonic, brightness, thickness, numSides);
        ctx.restore();

        // LAYER 1: Main spiral arms - medium speed, full reactivity
        ctx.save();
        ctx.rotate(this.accumulatedRotation * (0.4 + harmonic * 0.2));
        this.drawFractalLayer(ctx, arms, radius * orbitFactor * (0.8 + energy * 0.3), (hue + 30) % 360, seed + 100, 1, energy, harmonic, brightness, thickness, numSides);
        ctx.restore();

        // LAYER 2: Counter-rotating spirals - creates depth
        ctx.save();
        ctx.rotate(-this.accumulatedRotation * (0.6 + energy * 0.25));
        this.drawFractalLayer(ctx, arms + 2, radius * orbitFactor * (0.6 + harmonic * 0.2), (hue + 60) % 360, seed + 200, 2, energy, harmonic, brightness, thickness, numSides);
        ctx.restore();

        // LAYER 3: Fast inner spirals - very reactive
        ctx.save();
        ctx.rotate(this.accumulatedRotation * (1.0 + energy * 0.5));
        this.drawFractalLayer(ctx, arms * 2, radius * orbitFactor * (0.35 + energy * 0.15), (hue + 120) % 360, seed + 300, 3, energy, harmonic, brightness, thickness, numSides);
        ctx.restore();

        // Fractal sub-spirals at branch points
        this.drawFractalBranches(ctx, arms, radius * orbitFactor, hue, seed, energy, harmonic, brightness, thickness);

        // Reactive central vortex
        this.drawSpiralVortex(ctx, radius * orbitFactor * 0.25, hue, energy, harmonic, brightness);

        ctx.restore();
    }

    drawFractalLayer(ctx, arms, maxRadius, hue, seed, layer, energy, harmonic, brightness, thickness, numSides) {
        const config = this.config;

        // Spiral parameters that react to audio
        const spiralTurns = 2 + harmonic * 2 + brightness; // More turns with harmonic content
        const pointsPerArm = 60 + Math.floor(brightness * 40);
        const armWidth = thickness * (0.5 + energy * 1.5) * (1 - layer * 0.15); // PULSES with energy

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

                // Inner bright core — polygon shape from numSides
                const nodeRot = this.accumulatedRotation * 0.5 + n * 0.3;
                ctx.beginPath();
                for (let ns = 0; ns < numSides; ns++) {
                    const na = (Math.PI * 2 * ns) / numSides + nodeRot;
                    const nx = x + Math.cos(na) * nodeSize;
                    const ny = y + Math.sin(na) * nodeSize;
                    ns === 0 ? ctx.moveTo(nx, ny) : ctx.lineTo(nx, ny);
                }
                ctx.closePath();
                ctx.fillStyle = `hsla(${nodeHue}, ${config.saturation}%, ${75 + energy * 20}%, ${0.6 + energy * 0.4})`;
                ctx.fill();
            }
        }
    }

    drawFractalBranches(ctx, arms, maxRadius, hue, seed, energy, harmonic, brightness, thickness) {
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
                    ctx.lineWidth = thickness * (0.3 + energy * 0.6);
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

    /**
     * Circuit style - hexagonal grid with glowing circuit traces
     * Uses config.mirrors, radius, orbitRadius, rotationSpeed for dynamic control
     */
    renderCircuitStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const mirrors = config.mirrors;
        const orbitDist = config.orbitRadius * (0.6 + harmonic * 0.4);

        ctx.save();
        ctx.translate(centerX, centerY);

        // Global rotation
        ctx.rotate(this.accumulatedRotation * 0.15);

        // Hexagon size based on radius parameter
        const hexSize = radius * 0.25 * (0.8 + energy * 0.4);
        const rings = Math.max(3, Math.ceil(orbitDist / hexSize) + 1);

        // Draw in radial mirror segments for kaleidoscope effect
        for (let m = 0; m < mirrors; m++) {
            const mirrorAngle = (Math.PI * 2 * m) / mirrors;

            ctx.save();
            ctx.rotate(mirrorAngle);

            // Draw hexagonal circuit pattern in this segment
            for (let ring = 0; ring < rings; ring++) {
                const ringRadius = ring * hexSize * 1.5 + radius * 0.3;
                const hexPerRing = Math.max(1, Math.floor(ring * 1.5));
                const ringHue = (hue + ring * 20 + m * (360 / mirrors) + harmonic * 30) % 360;
                const ringAlpha = Math.max(0.15, 1 - ring * 0.12);

                for (let h = 0; h < hexPerRing; h++) {
                    const hexAngle = (h / hexPerRing) * (Math.PI * 2 / mirrors) - Math.PI / mirrors / 2;
                    const hx = ringRadius * Math.cos(hexAngle);
                    const hy = ringRadius * Math.sin(hexAngle);

                    // Only draw if in our wedge
                    if (Math.abs(hexAngle) > Math.PI / mirrors + 0.1) continue;

                    // Dynamic hex size
                    const dynamicHexSize = hexSize * (0.8 + brightness * 0.4);

                    // Draw polygon (sides from numSides knob)
                    const hexPoints = [];
                    for (let i = 0; i < numSides; i++) {
                        const a = (Math.PI * 2 / numSides) * i + Math.PI / numSides + this.accumulatedRotation * 0.5;
                        hexPoints.push({
                            x: hx + dynamicHexSize * 0.8 * Math.cos(a),
                            y: hy + dynamicHexSize * 0.8 * Math.sin(a)
                        });
                    }

                    // Polygon outline
                    ctx.beginPath();
                    ctx.moveTo(hexPoints[0].x, hexPoints[0].y);
                    for (let i = 1; i < numSides; i++) {
                        ctx.lineTo(hexPoints[i].x, hexPoints[i].y);
                    }
                    ctx.closePath();
                    ctx.strokeStyle = `hsla(${ringHue}, ${config.saturation}%, ${50 + energy * 30}%, ${ringAlpha * (0.4 + energy * 0.5)})`;
                    ctx.lineWidth = thickness * (0.5 + energy * 0.5);
                    ctx.stroke();

                    // Circuit traces
                    const nodeSeed = seed + ring * 100 + h * 7 + m;
                    if (this.seededRandom(nodeSeed) > 0.35) {
                        const traceCount = 1 + Math.floor(this.seededRandom(nodeSeed + 1) * 2);
                        for (let t = 0; t < traceCount; t++) {
                            const startIdx = Math.floor(this.seededRandom(nodeSeed + t * 10) * numSides);
                            const endIdx = (startIdx + Math.floor(numSides / 2)) % numSides;

                            ctx.beginPath();
                            ctx.moveTo(hexPoints[startIdx].x, hexPoints[startIdx].y);
                            ctx.lineTo(hx, hy);
                            ctx.lineTo(hexPoints[endIdx].x, hexPoints[endIdx].y);

                            const traceHue = (ringHue + t * 50) % 360;
                            ctx.strokeStyle = `hsla(${traceHue}, ${config.saturation}%, ${60 + brightness * 30}%, ${ringAlpha * (0.6 + energy * 0.4)})`;
                            ctx.lineWidth = thickness * (0.6 + energy * 0.6);
                            ctx.stroke();
                        }
                    }

                    // Glowing node
                    if (this.seededRandom(nodeSeed + 5) > 0.25) {
                        const nodeSize = (radius * 0.03 + energy * radius * 0.05) * (1 - ring * 0.08);
                        const pulsePhase = Math.sin(this.accumulatedRotation * 5 + ring * 0.5 + h * 0.3);
                        const nodePulse = 1 + pulsePhase * 0.4 * energy;

                        const glowGrad = ctx.createRadialGradient(hx, hy, 0, hx, hy, nodeSize * 3 * nodePulse);
                        glowGrad.addColorStop(0, `hsla(${ringHue}, ${config.saturation}%, 85%, ${0.7 + energy * 0.3})`);
                        glowGrad.addColorStop(0.5, `hsla(${ringHue}, ${config.saturation}%, 60%, ${0.25 + energy * 0.2})`);
                        glowGrad.addColorStop(1, `hsla(${ringHue}, ${config.saturation}%, 40%, 0)`);
                        ctx.fillStyle = glowGrad;
                        ctx.beginPath();
                        ctx.arc(hx, hy, nodeSize * 3 * nodePulse, 0, Math.PI * 2);
                        ctx.fill();

                        ctx.beginPath();
                        ctx.arc(hx, hy, nodeSize * nodePulse, 0, Math.PI * 2);
                        ctx.fillStyle = `hsla(${ringHue}, ${config.saturation * 0.4}%, 95%, 0.95)`;
                        ctx.fill();
                    }
                }
            }

            ctx.restore();
        }

        // Pulsing energy waves
        const waveCount = Math.max(2, Math.floor(mirrors / 3));
        for (let w = 0; w < waveCount; w++) {
            const wavePhase = (this.accumulatedRotation * 2 + w * (Math.PI * 2 / waveCount)) % (Math.PI * 2);
            const waveRadius = (wavePhase / (Math.PI * 2)) * orbitDist;
            const waveAlpha = 1 - wavePhase / (Math.PI * 2);

            ctx.beginPath();
            ctx.arc(0, 0, waveRadius, 0, Math.PI * 2);
            ctx.strokeStyle = `hsla(${hue}, ${config.saturation}%, 70%, ${waveAlpha * energy * 0.6})`;
            ctx.lineWidth = thickness * (0.8 + energy * 1.2);
            ctx.stroke();
        }

        // Central core - size based on radius
        const coreSize = radius * 0.4 * (0.8 + energy * 0.4);

        // Core polygon (sides from numSides knob)
        ctx.beginPath();
        for (let i = 0; i < numSides; i++) {
            const a = (Math.PI * 2 / numSides) * i + this.accumulatedRotation * 2;
            const x = coreSize * Math.cos(a);
            const y = coreSize * Math.sin(a);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = `hsla(${hue}, ${config.saturation}%, ${80 + energy * 20}%, ${0.85 + energy * 0.15})`;
        ctx.lineWidth = thickness * (1 + energy * 0.8);
        ctx.stroke();

        // Inner rotating polygon
        ctx.beginPath();
        for (let i = 0; i < numSides; i++) {
            const a = (Math.PI * 2 / numSides) * i - this.accumulatedRotation * 3;
            const x = coreSize * 0.5 * Math.cos(a);
            const y = coreSize * 0.5 * Math.sin(a);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = `hsla(${(hue + 60) % 360}, ${config.saturation}%, 75%, ${0.7 + energy * 0.3})`;
        ctx.lineWidth = thickness * (0.7 + energy * 0.5);
        ctx.stroke();

        // Bright core
        const coreGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, coreSize * 0.4);
        coreGrad.addColorStop(0, `hsla(${hue}, ${config.saturation * 0.4}%, 95%, 1)`);
        coreGrad.addColorStop(0.5, `hsla(${hue}, ${config.saturation}%, 70%, 0.6)`);
        coreGrad.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 50%, 0)`);
        ctx.fillStyle = coreGrad;
        ctx.beginPath();
        ctx.arc(0, 0, coreSize * 0.4, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    /**
     * Fibonacci style - golden spiral meets kaleidoscope
     * Expresses the beauty of the golden ratio through phyllotaxis patterns,
     * reflected spiral geometry, and sacred proportions
     */
    renderFibonacciStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const mirrors = config.mirrors;
        const orbitDist = config.orbitRadius * (0.7 + harmonic * 0.4);
        const rotSpeed = config.rotationSpeed;

        const PHI = 1.618033988749895;
        const GOLDEN_ANGLE = Math.PI * 2 / (PHI * PHI); // ~137.5 degrees

        ctx.save();
        ctx.translate(centerX, centerY);

        // --- Layer 0: Kaleidoscopic reflected golden spirals ---
        // Draw a pair of spirals per mirror segment for reflected symmetry
        for (let m = 0; m < mirrors; m++) {
            const mirrorAngle = (Math.PI * 2 * m) / mirrors;

            ctx.save();
            ctx.rotate(mirrorAngle + this.accumulatedRotation * 0.3);

            // Primary golden spiral arm
            this.drawGoldenSpiral(ctx, radius * (0.85 + energy * 0.15),
                (hue + m * (360 / mirrors)) % 360, energy, brightness, harmonic,
                seed + m * 100, radius, thickness, rotSpeed);

            // Counter-spiral (reflected) for kaleidoscope effect
            ctx.save();
            ctx.scale(-1, 1);
            this.drawGoldenSpiral(ctx, radius * (0.65 + harmonic * 0.2),
                (hue + m * (360 / mirrors) + 40) % 360, energy * 0.7, brightness, harmonic,
                seed + m * 100 + 50, radius * 0.8, thickness * 0.7, rotSpeed * 0.8);
            ctx.restore();

            ctx.restore();
        }

        // --- Layer 1: Phyllotaxis sunflower pattern (the heart of Fibonacci) ---
        // Elements placed at golden angle intervals, distance = sqrt(n)
        const phyllotaxisCount = 80 + Math.floor(brightness * 40 + energy * 30);
        const phyllotaxisScale = radius * 0.7 * (0.8 + energy * 0.25);

        for (let n = 1; n < phyllotaxisCount; n++) {
            const angle = n * GOLDEN_ANGLE + this.accumulatedRotation * 0.4;
            const dist = Math.sqrt(n) / Math.sqrt(phyllotaxisCount) * phyllotaxisScale;

            const x = dist * Math.cos(angle);
            const y = dist * Math.sin(angle);

            const distRatio = dist / phyllotaxisScale;
            const nHue = (hue + n * 2.5 + harmonic * 50) % 360;

            // Size follows Fibonacci-like growth: larger toward edge
            const baseNodeSize = (2 + distRatio * 8) * (0.7 + energy * 0.6);
            const pulsePhase = Math.sin(this.accumulatedRotation * 3 - dist * 0.015 + n * 0.1);
            const nodeSize = baseNodeSize * (0.85 + pulsePhase * 0.15 * energy);

            // Outer glow for larger nodes
            if (nodeSize > 4) {
                ctx.beginPath();
                ctx.arc(x, y, nodeSize * 2.5, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${nHue}, ${config.saturation}%, 70%, ${0.06 + energy * 0.1})`;
                ctx.fill();
            }

            // Main element - alternate between circles and tiny polygons
            if (n % 3 === 0) {
                // Small gem (sides from numSides knob)
                const sides = numSides;
                ctx.beginPath();
                for (let s = 0; s < sides; s++) {
                    const sa = (Math.PI * 2 * s) / sides + angle * 0.3;
                    const px = x + Math.cos(sa) * nodeSize;
                    const py = y + Math.sin(sa) * nodeSize;
                    s === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
                }
                ctx.closePath();
                ctx.fillStyle = `hsla(${nHue}, ${config.saturation}%, ${55 + energy * 20 + brightness * 15}%, ${0.35 + energy * 0.35})`;
                ctx.fill();
                ctx.strokeStyle = `hsla(${nHue}, ${config.saturation * 0.8}%, ${70 + energy * 20}%, ${0.25 + energy * 0.3})`;
                ctx.lineWidth = 0.5 + energy * 0.5;
                ctx.stroke();
            } else {
                // Circle
                ctx.beginPath();
                ctx.arc(x, y, nodeSize, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${nHue}, ${config.saturation}%, ${60 + brightness * 20 + energy * 15}%, ${0.3 + energy * 0.4})`;
                ctx.fill();
            }
        }

        // --- Layer 2: Nested golden rectangles with arc connections ---
        ctx.save();
        ctx.rotate(this.accumulatedRotation * 0.15);
        let rectSize = orbitDist * 0.8 * (0.75 + energy * 0.35);
        let rectRotation = 0;

        const rectCount = Math.min(12, Math.max(5, mirrors + 2));
        for (let r = 0; r < rectCount; r++) {
            const rectHue = (hue + r * (150 / rectCount) + brightness * 25) % 360;
            const alpha = (0.45 - r * 0.03 + energy * 0.35) * (1 - r / rectCount * 0.3);

            ctx.save();
            ctx.rotate(rectRotation);

            const w = rectSize;
            const h = rectSize / PHI;

            // Golden rectangle
            ctx.beginPath();
            ctx.rect(-w / 2, -h / 2, w, h);
            ctx.strokeStyle = `hsla(${rectHue}, ${config.saturation}%, ${55 + energy * 25}%, ${alpha})`;
            ctx.lineWidth = thickness * (0.5 + energy * 0.5);
            ctx.stroke();

            // Golden spiral arc within rectangle
            ctx.beginPath();
            const arcRadius = h;
            // Draw quarter-circle arc connecting rectangle corners
            ctx.arc(-w / 2 + h, -h / 2, arcRadius, Math.PI, Math.PI * 1.5);
            ctx.strokeStyle = `hsla(${(rectHue + 30) % 360}, ${config.saturation}%, ${65 + energy * 25}%, ${alpha * 0.8})`;
            ctx.lineWidth = thickness * (0.4 + energy * 0.6);
            ctx.stroke();

            ctx.restore();

            rectSize /= PHI;
            rectRotation += Math.PI / 2;
        }
        ctx.restore();

        // --- Layer 3: Fibonacci-numbered petal rings ---
        const fibRings = [3, 5, 8, 13];
        for (let fi = 0; fi < fibRings.length; fi++) {
            const petalCount = fibRings[fi];
            const ringRadius = radius * (0.25 + fi * 0.22) * (0.8 + energy * 0.3);
            const petalLength = radius * (0.15 + fi * 0.04) * (0.7 + energy * 0.5);
            const petalWidth = petalLength * (0.2 + brightness * 0.1) / PHI;
            const ringRotation = this.accumulatedRotation * (0.6 - fi * 0.1) * (fi % 2 === 0 ? 1 : -1);

            for (let p = 0; p < petalCount; p++) {
                const petalAngle = (Math.PI * 2 * p) / petalCount + ringRotation;
                const petalHue = (hue + p * (360 / petalCount) + fi * 25 + harmonic * 30) % 360;

                ctx.save();
                ctx.rotate(petalAngle);
                ctx.translate(ringRadius, 0);
                ctx.rotate(petalAngle * 0.3);

                // Petal with golden ratio proportions
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.bezierCurveTo(
                    petalWidth, petalLength * 0.35,
                    petalWidth * 0.6, petalLength * 0.75,
                    0, petalLength
                );
                ctx.bezierCurveTo(
                    -petalWidth * 0.6, petalLength * 0.75,
                    -petalWidth, petalLength * 0.35,
                    0, 0
                );

                const alpha = 0.2 + energy * 0.3 - fi * 0.03;
                ctx.fillStyle = `hsla(${petalHue}, ${config.saturation}%, ${55 + energy * 25}%, ${alpha})`;
                ctx.fill();
                ctx.strokeStyle = `hsla(${petalHue}, ${config.saturation}%, ${72 + energy * 18}%, ${alpha + 0.2})`;
                ctx.lineWidth = thickness * (0.3 + energy * 0.4);
                ctx.stroke();

                ctx.restore();
            }
        }

        // --- Central sacred geometry core ---
        // Vesica piscis / seed of life pattern
        const coreSize = radius * 0.18 * (0.85 + energy * 0.4);
        const coreCircles = numSides;
        for (let c = 0; c < coreCircles; c++) {
            const cAngle = (Math.PI * 2 * c) / coreCircles + this.accumulatedRotation * 0.5;
            const cx = Math.cos(cAngle) * coreSize * 0.5;
            const cy = Math.sin(cAngle) * coreSize * 0.5;
            const cHue = (hue + c * 60 + harmonic * 30) % 360;

            ctx.beginPath();
            ctx.arc(cx, cy, coreSize * 0.55, 0, Math.PI * 2);
            ctx.strokeStyle = `hsla(${cHue}, ${config.saturation}%, ${70 + energy * 20}%, ${0.3 + energy * 0.3})`;
            ctx.lineWidth = thickness * (0.3 + energy * 0.3);
            ctx.stroke();
        }

        // Central glow
        const coreGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, coreSize * 1.5);
        coreGrad.addColorStop(0, `hsla(${(hue + 30) % 360}, ${config.saturation * 0.5}%, 95%, ${0.9 + energy * 0.1})`);
        coreGrad.addColorStop(0.4, `hsla(${hue}, ${config.saturation}%, 75%, ${0.4 + energy * 0.3})`);
        coreGrad.addColorStop(1, `hsla(${hue}, ${config.saturation}%, 50%, 0)`);
        ctx.fillStyle = coreGrad;
        ctx.beginPath();
        ctx.arc(0, 0, coreSize * 1.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    drawGoldenSpiral(ctx, maxRadius, hue, energy, brightness, harmonic, seed, baseRadius, thickness, rotSpeed) {
        const config = this.config;
        const PHI = 1.618033988749895;
        const points = 140;

        const radiusScale = baseRadius * 0.03;

        // Draw the golden spiral path
        ctx.beginPath();
        let prevX, prevY;
        for (let i = 0; i < points; i++) {
            const t = i / points;
            const angle = t * Math.PI * 5 * rotSpeed;
            const spiralRadius = Math.pow(PHI, angle / (Math.PI / 2)) * radiusScale * (0.75 + energy * 0.35);

            if (spiralRadius > maxRadius) break;

            // Add organic breathing modulation
            const breathe = Math.sin(this.accumulatedRotation * 1.5 + t * Math.PI * 2) * harmonic * spiralRadius * 0.04;
            const r = spiralRadius + breathe;

            const x = r * Math.cos(angle);
            const y = r * Math.sin(angle);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);

            prevX = x;
            prevY = y;
        }

        const spiralHue = (hue + brightness * 35) % 360;
        ctx.strokeStyle = `hsla(${spiralHue}, ${config.saturation}%, ${58 + energy * 25}%, ${0.45 + energy * 0.4})`;
        ctx.lineWidth = thickness * (0.6 + energy * 1.2);
        ctx.lineCap = 'round';
        ctx.stroke();

        // Nodes at Fibonacci intervals along the spiral
        const fibPositions = [1, 2, 3, 5, 8, 13, 21, 34, 55];
        for (let fi = 0; fi < fibPositions.length; fi++) {
            const i = Math.min(points - 1, fibPositions[fi] * 2);
            const t = i / points;
            const angle = t * Math.PI * 5 * rotSpeed;
            const spiralRadius = Math.pow(PHI, angle / (Math.PI / 2)) * radiusScale * (0.75 + energy * 0.35);

            if (spiralRadius > maxRadius) break;

            const x = spiralRadius * Math.cos(angle);
            const y = spiralRadius * Math.sin(angle);

            const nodeSize = (baseRadius * 0.012 + thickness * 0.25) * (0.7 + energy * 0.7) * (0.6 + fi * 0.08);
            const nodeHue = (spiralHue + fi * 15 + harmonic * 30) % 360;

            // Glow
            ctx.beginPath();
            ctx.arc(x, y, nodeSize * 2.5, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${nodeHue}, ${config.saturation}%, 70%, ${0.08 + energy * 0.15})`;
            ctx.fill();

            // Core node
            ctx.beginPath();
            ctx.arc(x, y, nodeSize, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${nodeHue}, ${config.saturation}%, ${78 + energy * 18}%, ${0.6 + energy * 0.35})`;
            ctx.fill();
        }
    }

    /**
     * DMT style — "Hyperspace" recursive geometry
     * Hyper-grid mesh, chrysanthemum cycloids, Penrose-like tiling, impossible solids,
     * Lissajous breathing center, neon-saturated palette
     */
    renderDMTStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const mirrors = config.mirrors;
        const rot = this.accumulatedRotation;

        // Lissajous breathing offset for the "eye"
        const breatheX = Math.sin(rot * 0.7) * radius * 0.06 * (1 + energy * 0.5);
        const breatheY = Math.cos(rot * 1.1) * radius * 0.06 * (1 + energy * 0.5);
        const eyeX = centerX + breatheX;
        const eyeY = centerY + breatheY;

        // Segment count oscillates with complexity (6-24 mapped from mirrors)
        const segmentCount = Math.max(6, Math.min(24, mirrors * 2 + Math.floor(brightness * 8)));

        // Seed-based variation
        const rotDir = this.seededRandom(seed) > 0.5 ? 1 : -1;
        const layerDepth = 2 + Math.floor(this.seededRandom(seed + 1) * 3);
        const hueShift = this.seededRandom(seed + 2) * 80;
        const hasImpossible = this.seededRandom(seed + 3) > 0.35;

        // --- Hyper-Grid: Curved wireframe mesh toward singularity ---
        for (let i = 0; i < segmentCount; i++) {
            const segAngle = (Math.PI * 2 * i) / segmentCount + rot * 0.2 * rotDir;

            ctx.save();
            ctx.translate(eyeX, eyeY);
            ctx.rotate(segAngle);

            // Radial grid lines curving inward
            const gridLines = Math.max(3, Math.floor(numSides / 2)) + Math.floor(energy * 3);
            for (let g = 0; g < gridLines; g++) {
                const gRatio = (g + 1) / gridLines;
                const startR = radius * 0.15 * gRatio;
                const endR = radius * (0.6 + gRatio * 0.4) * (0.9 + energy * 0.2);
                const curve = Math.sin(rot * 1.5 + g) * radius * 0.08 * harmonic;
                const gridHue = (hue + g * 30 + hueShift) % 360;

                ctx.beginPath();
                ctx.moveTo(startR, 0);
                ctx.quadraticCurveTo(
                    (startR + endR) / 2, curve,
                    endR, 0
                );
                ctx.strokeStyle = `hsla(${gridHue}, ${Math.min(100, config.saturation * 1.2)}%, ${55 + energy * 25}%, ${0.3 + energy * 0.35})`;
                ctx.lineWidth = thickness * (0.3 + gRatio * 0.4);
                ctx.stroke();
            }

            ctx.restore();
        }

        // Concentric arc rings (cross-grid)
        const arcRings = 5 + Math.floor(energy * 3);
        for (let r = 0; r < arcRings; r++) {
            const arcR = radius * (0.2 + r * 0.15) * (0.85 + energy * 0.2);
            const arcHue = (hue + r * 25 + brightness * 30 + hueShift) % 360;
            const wobble = Math.sin(rot * 2 + r * 0.8) * 0.03 * energy;

            ctx.beginPath();
            ctx.arc(eyeX, eyeY, arcR * (1 + wobble), 0, Math.PI * 2);
            ctx.strokeStyle = `hsla(${arcHue}, ${Math.min(100, config.saturation * 1.1)}%, ${50 + energy * 20}%, ${0.2 + energy * 0.25 - r * 0.02})`;
            ctx.lineWidth = thickness * (0.4 + energy * 0.3);
            ctx.stroke();
        }

        // --- Chrysanthemum: Dense cycloid flower at each mirror ---
        for (let m = 0; m < mirrors; m++) {
            const mirrorAngle = (Math.PI * 2 * m) / mirrors + rot * 0.3 * rotDir;
            const orbitDist = config.orbitRadius * (0.5 + harmonic * 0.5);
            const mx = eyeX + Math.cos(mirrorAngle) * orbitDist;
            const my = eyeY + Math.sin(mirrorAngle) * orbitDist;

            // Cycloid petals
            const petalCount = 8 + Math.floor(this.seededRandom(seed + 10 + m) * 8);
            const petalRadius = radius * 0.25 * (0.7 + this.seededRandom(seed + 11 + m) * 0.5);

            ctx.save();
            ctx.translate(mx, my);
            ctx.rotate(mirrorAngle + rot * rotDir);

            for (let p = 0; p < petalCount; p++) {
                const pAngle = (Math.PI * 2 * p) / petalCount;
                const petalHue = (hue + p * (360 / petalCount) + hueShift) % 360;
                const pLen = petalRadius * (0.6 + energy * 0.5 + Math.sin(rot * 3 + p) * 0.15);

                ctx.beginPath();
                ctx.moveTo(0, 0);
                // Cycloid-like curve
                const cp1x = Math.cos(pAngle - 0.4) * pLen * 0.5;
                const cp1y = Math.sin(pAngle - 0.4) * pLen * 0.5;
                const cp2x = Math.cos(pAngle + 0.2) * pLen * 0.9;
                const cp2y = Math.sin(pAngle + 0.2) * pLen * 0.9;
                const endX = Math.cos(pAngle) * pLen;
                const endY = Math.sin(pAngle) * pLen;
                ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, endX, endY);

                ctx.strokeStyle = `hsla(${petalHue}, ${Math.min(100, config.saturation * 1.3)}%, ${60 + energy * 25}%, ${0.35 + energy * 0.4})`;
                ctx.lineWidth = thickness * (0.4 + energy * 0.3);
                ctx.lineCap = 'round';
                ctx.stroke();
            }

            ctx.restore();
        }

        // --- Impossible Solids: Hyper-cube wireframes at mirror positions ---
        if (hasImpossible) {
            for (let m = 0; m < mirrors; m++) {
                const mirrorAngle = (Math.PI * 2 * m) / mirrors + rot * 0.15 * rotDir;
                const dist = config.orbitRadius * 0.7;
                const ix = eyeX + Math.cos(mirrorAngle) * dist;
                const iy = eyeY + Math.sin(mirrorAngle) * dist;
                const cubeSize = radius * 0.12 * (0.8 + this.seededRandom(seed + 20 + m) * 0.4);
                const innerOffset = cubeSize * 0.5;
                const cubeRot = rot * (1.2 + this.seededRandom(seed + 21 + m)) * rotDir;
                const cubeHue = (hue + 120 + m * 30) % 360;

                ctx.save();
                ctx.translate(ix, iy);
                ctx.rotate(cubeRot);

                // Outer square
                ctx.beginPath();
                ctx.rect(-cubeSize, -cubeSize, cubeSize * 2, cubeSize * 2);
                ctx.strokeStyle = `hsla(${cubeHue}, ${config.saturation}%, ${65 + energy * 20}%, ${0.3 + energy * 0.35})`;
                ctx.lineWidth = thickness * 0.5;
                ctx.stroke();

                // Inner square (offset for 3D illusion)
                ctx.beginPath();
                ctx.rect(-cubeSize + innerOffset, -cubeSize + innerOffset, cubeSize * 2 - innerOffset * 2, cubeSize * 2 - innerOffset * 2);
                ctx.strokeStyle = `hsla(${(cubeHue + 60) % 360}, ${config.saturation}%, ${55 + energy * 20}%, ${0.25 + energy * 0.3})`;
                ctx.lineWidth = thickness * 0.4;
                ctx.stroke();

                // Connecting edges (impossible perspective)
                const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
                for (const [cx, cy] of corners) {
                    ctx.beginPath();
                    ctx.moveTo(cx * cubeSize, cy * cubeSize);
                    ctx.lineTo(cx * (cubeSize - innerOffset), cy * (cubeSize - innerOffset));
                    ctx.strokeStyle = `hsla(${(cubeHue + 30) % 360}, ${config.saturation * 0.9}%, ${60 + energy * 15}%, ${0.2 + energy * 0.25})`;
                    ctx.lineWidth = thickness * 0.35;
                    ctx.stroke();
                }

                ctx.restore();
            }
        }

        // --- Aperiodic tile accents: Penrose-like kite/dart shapes ---
        for (let layer = 0; layer < layerDepth; layer++) {
            const layerR = radius * (0.3 + layer * 0.25);
            const tileCount = segmentCount;
            const tileHue = (hue + layer * 50 + hueShift) % 360;

            for (let t = 0; t < tileCount; t++) {
                const tAngle = (Math.PI * 2 * t) / tileCount + rot * (0.1 + layer * 0.05) * rotDir;
                const tx = eyeX + Math.cos(tAngle) * layerR;
                const ty = eyeY + Math.sin(tAngle) * layerR;
                const tSize = radius * 0.06 * (0.8 + energy * 0.4);

                // Kite shape (Penrose-like)
                ctx.save();
                ctx.translate(tx, ty);
                ctx.rotate(tAngle + rot * 0.5);

                ctx.beginPath();
                ctx.moveTo(0, -tSize * 1.618);
                ctx.lineTo(tSize * 0.6, 0);
                ctx.lineTo(0, tSize * 0.5);
                ctx.lineTo(-tSize * 0.6, 0);
                ctx.closePath();

                ctx.strokeStyle = `hsla(${tileHue}, ${Math.min(100, config.saturation * 1.2)}%, ${55 + energy * 25}%, ${0.25 + energy * 0.3 - layer * 0.04})`;
                ctx.lineWidth = thickness * (0.3 + energy * 0.2);
                ctx.stroke();

                ctx.restore();
            }
        }

        // Central eye — iridescent core
        const eyeRadius = radius * 0.15 * (0.8 + energy * 0.5);
        const eyeGrad = ctx.createRadialGradient(eyeX, eyeY, 0, eyeX, eyeY, eyeRadius);
        eyeGrad.addColorStop(0, `hsla(${(hue + rot * 20) % 360}, ${Math.min(100, config.saturation * 1.3)}%, 70%, ${0.5 + energy * 0.4})`);
        eyeGrad.addColorStop(0.5, `hsla(${(hue + 90 + rot * 20) % 360}, ${config.saturation}%, 55%, ${0.3 + energy * 0.3})`);
        eyeGrad.addColorStop(1, `hsla(${(hue + 180 + rot * 20) % 360}, ${config.saturation}%, 40%, 0)`);
        ctx.beginPath();
        ctx.arc(eyeX, eyeY, eyeRadius, 0, Math.PI * 2);
        ctx.fillStyle = eyeGrad;
        ctx.fill();
    }

    /**
     * Sacred Geometry style — "Cloud Chamber / Radioactive Decay"
     * Alpha/beta/gamma particle tracks, cloud chamber physics, Cherenkov glow,
     * decay curves, snowflake kaleidoscope from chaotic tracks
     */
    renderSacredStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const mirrors = config.mirrors;
        const rot = this.accumulatedRotation;
        const orbitFactor = config.orbitRadius / 200;

        // Seed-based variation
        const rotDir = this.seededRandom(seed) > 0.5 ? 1 : -1;
        const hueShift = this.seededRandom(seed + 2) * 40;
        const density = numSides + Math.floor(this.seededRandom(seed + 1) * 4);

        // Archaic palette: user hue as base, offsets for secondary/tertiary
        const amberHue = hue;
        const crimsonHue = (hue - 45 + 360) % 360;
        const boneHue = (hue + 10) % 360;

        // --- Kaleidoscope mirrored symbol rings ---
        for (let m = 0; m < mirrors; m++) {
            const mirrorAngle = (Math.PI * 2 * m) / mirrors + rot * 0.12 * rotDir;

            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(mirrorAngle);
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            // Inner ring: large slow-rotating sigils
            const innerCount = density;
            for (let i = 0; i < innerCount; i++) {
                const iSeed = seed + 100 + m * 30 + i * 7;
                const iAngle = (this.seededRandom(iSeed) - 0.5) * (Math.PI / mirrors) * 0.8;
                const iDist = radius * orbitFactor * (0.1 + this.seededRandom(iSeed + 1) * 0.2);
                const iSize = radius * (0.06 + this.seededRandom(iSeed + 2) * 0.05) * (0.8 + energy * 0.4);
                const iType = Math.floor(this.seededRandom(iSeed + 3) * 16);
                const iHue = (amberHue + this.seededRandom(iSeed + 4) * 30 + hueShift) % 360;
                const ix = Math.cos(iAngle) * iDist;
                const iy = Math.sin(iAngle) * iDist;
                const iRot = rot * 0.3 * rotDir + this.seededRandom(iSeed + 5) * Math.PI;

                ctx.save();
                ctx.translate(ix, iy);
                ctx.rotate(iRot);

                // Glow aura behind large symbols
                const glowR = iSize * 1.5;
                const glow = ctx.createRadialGradient(0, 0, 0, 0, 0, glowR);
                glow.addColorStop(0, `hsla(${iHue}, ${config.saturation * 0.6}%, 55%, ${0.08 + energy * 0.12})`);
                glow.addColorStop(1, `hsla(${iHue}, ${config.saturation * 0.4}%, 40%, 0)`);
                ctx.beginPath();
                ctx.arc(0, 0, glowR, 0, Math.PI * 2);
                ctx.fillStyle = glow;
                ctx.fill();

                ctx.strokeStyle = `hsla(${iHue}, ${config.saturation * 0.7}%, ${55 + energy * 20}%, ${0.4 + energy * 0.35})`;
                ctx.lineWidth = thickness * (0.6 + energy * 0.4);
                this._drawOccultSymbol(ctx, iType, iSize);
                ctx.restore();
            }

            // Middle ring: denser medium symbols
            const midCount = density + 3;
            for (let i = 0; i < midCount; i++) {
                const mSeed = seed + 300 + m * 30 + i * 11;
                const mAngle = (this.seededRandom(mSeed) - 0.5) * (Math.PI / mirrors);
                const mDist = radius * orbitFactor * (0.25 + this.seededRandom(mSeed + 1) * 0.3);
                const mSize = radius * (0.03 + this.seededRandom(mSeed + 2) * 0.04) * (0.7 + harmonic * 0.5);
                const mType = Math.floor(this.seededRandom(mSeed + 3) * 16);
                const mHue = (crimsonHue + this.seededRandom(mSeed + 4) * 50 - 25 + hueShift) % 360;
                const mx = Math.cos(mAngle) * mDist;
                const my = Math.sin(mAngle) * mDist;
                const mRot = rot * 0.5 * -rotDir + this.seededRandom(mSeed + 5) * Math.PI * 2;

                ctx.save();
                ctx.translate(mx, my);
                ctx.rotate(mRot);
                ctx.strokeStyle = `hsla(${mHue}, ${config.saturation * 0.6}%, ${50 + brightness * 15}%, ${0.3 + harmonic * 0.3})`;
                ctx.lineWidth = thickness * (0.4 + harmonic * 0.3);
                this._drawOccultSymbol(ctx, mType, mSize);
                ctx.restore();
            }

            // Outer ring: overwhelming swarm of tiny symbols
            const outerCount = density + 5 + Math.floor(energy * 4);
            for (let i = 0; i < outerCount; i++) {
                const oSeed = seed + 600 + m * 30 + i * 13;
                const oAngle = (this.seededRandom(oSeed) - 0.5) * (Math.PI / mirrors);
                const oDist = radius * orbitFactor * (0.5 + this.seededRandom(oSeed + 1) * 0.4);
                const oSize = radius * (0.015 + this.seededRandom(oSeed + 2) * 0.025) * (0.6 + brightness * 0.6);
                const oType = Math.floor(this.seededRandom(oSeed + 3) * 16);
                const oHue = (boneHue + this.seededRandom(oSeed + 4) * 40 + hueShift) % 360;
                const ox = Math.cos(oAngle) * oDist;
                const oy = Math.sin(oAngle) * oDist;
                const oRot = rot * 0.8 * rotDir + this.seededRandom(oSeed + 5) * Math.PI * 2;

                ctx.save();
                ctx.translate(ox, oy);
                ctx.rotate(oRot);
                ctx.strokeStyle = `hsla(${oHue}, ${config.saturation * 0.5}%, ${45 + brightness * 20}%, ${0.2 + brightness * 0.3})`;
                ctx.lineWidth = thickness * (0.25 + brightness * 0.2);
                this._drawOccultSymbol(ctx, oType, oSize);
                ctx.restore();
            }

            // Connecting filaments between symbols (arcane web)
            if (harmonic > 0.2) {
                const webCount = 3 + Math.floor(harmonic * 4);
                for (let w = 0; w < webCount; w++) {
                    const wSeed = seed + 900 + m * 20 + w;
                    const a1 = (this.seededRandom(wSeed) - 0.5) * (Math.PI / mirrors);
                    const d1 = radius * (0.1 + this.seededRandom(wSeed + 1) * 0.6);
                    const a2 = (this.seededRandom(wSeed + 2) - 0.5) * (Math.PI / mirrors);
                    const d2 = radius * (0.1 + this.seededRandom(wSeed + 3) * 0.6);
                    const wHue = (amberHue + this.seededRandom(wSeed + 4) * 30) % 360;

                    ctx.beginPath();
                    ctx.moveTo(Math.cos(a1) * d1, Math.sin(a1) * d1);
                    const cpx = Math.cos((a1 + a2) / 2) * (d1 + d2) * 0.3;
                    const cpy = Math.sin((a1 + a2) / 2) * (d1 + d2) * 0.3;
                    ctx.quadraticCurveTo(cpx, cpy, Math.cos(a2) * d2, Math.sin(a2) * d2);
                    ctx.strokeStyle = `hsla(${wHue}, ${config.saturation * 0.3}%, 40%, ${0.06 + harmonic * 0.1})`;
                    ctx.lineWidth = 0.4 + harmonic * 0.3;
                    ctx.stroke();
                }
            }

            ctx.restore();
        }

        // --- Central all-seeing eye / vortex ---
        const eyeR = radius * 0.1 * (0.7 + energy * 0.5);

        // Outer glow ring
        const eyeGlow = ctx.createRadialGradient(centerX, centerY, eyeR * 0.5, centerX, centerY, eyeR * 3);
        eyeGlow.addColorStop(0, `hsla(${amberHue + hueShift}, ${config.saturation * 0.7}%, 50%, ${0.1 + energy * 0.15})`);
        eyeGlow.addColorStop(1, `hsla(${crimsonHue}, ${config.saturation * 0.4}%, 30%, 0)`);
        ctx.beginPath();
        ctx.arc(centerX, centerY, eyeR * 3, 0, Math.PI * 2);
        ctx.fillStyle = eyeGlow;
        ctx.fill();

        // Eye shape (almond)
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.15 * rotDir);

        ctx.beginPath();
        ctx.moveTo(-eyeR * 1.6, 0);
        ctx.quadraticCurveTo(0, -eyeR * 1.0, eyeR * 1.6, 0);
        ctx.quadraticCurveTo(0, eyeR * 1.0, -eyeR * 1.6, 0);
        ctx.closePath();
        ctx.strokeStyle = `hsla(${amberHue + hueShift}, ${config.saturation * 0.8}%, ${55 + energy * 20}%, ${0.5 + energy * 0.35})`;
        ctx.lineWidth = thickness * (0.7 + energy * 0.5);
        ctx.stroke();

        // Iris
        ctx.beginPath();
        ctx.arc(0, 0, eyeR * 0.5, 0, Math.PI * 2);
        const irisGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, eyeR * 0.5);
        irisGrad.addColorStop(0, `hsla(${crimsonHue + hueShift}, ${config.saturation}%, ${40 + energy * 25}%, ${0.5 + energy * 0.4})`);
        irisGrad.addColorStop(0.7, `hsla(${amberHue + hueShift}, ${config.saturation * 0.8}%, 35%, ${0.3 + energy * 0.25})`);
        irisGrad.addColorStop(1, `hsla(0, 0%, 8%, ${0.4 + energy * 0.3})`);
        ctx.fillStyle = irisGrad;
        ctx.fill();

        // Pupil
        ctx.beginPath();
        ctx.arc(0, 0, eyeR * 0.15, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(0, 0%, 2%, ${0.6 + energy * 0.3})`;
        ctx.fill();

        ctx.restore();
    }

    /**
     * Helper: Draw one of 16 occult/alchemical symbols at origin
     * Symbols: pentagram, hexagram, triangle variants, planetary glyphs,
     * alchemical signs, runes, sigils, eye, ouroboros, infinity
     */
    _drawOccultSymbol(ctx, type, size) {
        const s = size;
        const PI = Math.PI;
        ctx.beginPath();
        switch (type) {
            case 0: // Pentagram (5-pointed star)
                for (let i = 0; i < 5; i++) {
                    const a = -PI / 2 + (i * 2 * PI * 2) / 5;
                    const x = Math.cos(a) * s;
                    const y = Math.sin(a) * s;
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                ctx.closePath();
                ctx.stroke();
                // Enclosing circle
                ctx.beginPath();
                ctx.arc(0, 0, s * 1.05, 0, PI * 2);
                ctx.stroke();
                break;
            case 1: // Hexagram (Star of David / Seal of Solomon)
                // Up triangle
                for (let i = 0; i < 3; i++) {
                    const a = -PI / 2 + (i * PI * 2) / 3;
                    const x = Math.cos(a) * s;
                    const y = Math.sin(a) * s;
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                ctx.closePath();
                ctx.stroke();
                // Down triangle
                ctx.beginPath();
                for (let i = 0; i < 3; i++) {
                    const a = PI / 2 + (i * PI * 2) / 3;
                    const x = Math.cos(a) * s;
                    const y = Math.sin(a) * s;
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                ctx.closePath();
                ctx.stroke();
                break;
            case 2: // Fire triangle (up)
                ctx.moveTo(0, -s);
                ctx.lineTo(s * 0.85, s * 0.7);
                ctx.lineTo(-s * 0.85, s * 0.7);
                ctx.closePath();
                ctx.stroke();
                break;
            case 3: // Water triangle (down)
                ctx.moveTo(0, s);
                ctx.lineTo(s * 0.85, -s * 0.7);
                ctx.lineTo(-s * 0.85, -s * 0.7);
                ctx.closePath();
                ctx.stroke();
                break;
            case 4: // Sun / Gold (circle with dot)
                ctx.arc(0, 0, s, 0, PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(0, 0, s * 0.15, 0, PI * 2);
                ctx.stroke();
                break;
            case 5: // Crescent Moon / Silver
                ctx.arc(0, 0, s, PI * 0.3, PI * 1.7);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(s * 0.3, 0, s * 0.75, PI * 0.35, PI * 1.65, true);
                ctx.stroke();
                break;
            case 6: // Mercury (circle + cross + horns)
                ctx.arc(0, 0, s * 0.5, 0, PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, s * 0.5);
                ctx.lineTo(0, s);
                ctx.moveTo(-s * 0.35, s * 0.75);
                ctx.lineTo(s * 0.35, s * 0.75);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(0, -s * 0.5, s * 0.4, PI * 1.2, PI * 1.8);
                ctx.stroke();
                break;
            case 7: // Venus / Copper (circle + cross below)
                ctx.arc(0, -s * 0.2, s * 0.45, 0, PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, s * 0.25);
                ctx.lineTo(0, s);
                ctx.moveTo(-s * 0.3, s * 0.6);
                ctx.lineTo(s * 0.3, s * 0.6);
                ctx.stroke();
                break;
            case 8: // Mars / Iron (circle + arrow)
                ctx.arc(-s * 0.15, s * 0.15, s * 0.5, 0, PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(s * 0.2, -s * 0.2);
                ctx.lineTo(s * 0.7, -s * 0.7);
                ctx.lineTo(s * 0.7, -s * 0.35);
                ctx.moveTo(s * 0.7, -s * 0.7);
                ctx.lineTo(s * 0.35, -s * 0.7);
                ctx.stroke();
                break;
            case 9: // Saturn / Lead (cross + curve)
                ctx.moveTo(0, -s * 0.8);
                ctx.lineTo(0, s * 0.6);
                ctx.moveTo(-s * 0.35, -s * 0.3);
                ctx.lineTo(s * 0.35, -s * 0.3);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(s * 0.25, -s * 0.6, s * 0.35, PI * 0.5, PI * 1.5, true);
                ctx.stroke();
                break;
            case 10: // Triquetra (three interlocking arcs)
                for (let i = 0; i < 3; i++) {
                    const a = -PI / 2 + (i * PI * 2) / 3;
                    const cx = Math.cos(a) * s * 0.35;
                    const cy = Math.sin(a) * s * 0.35;
                    ctx.beginPath();
                    ctx.arc(cx, cy, s * 0.6, a - PI * 0.4, a + PI * 0.4);
                    ctx.stroke();
                }
                break;
            case 11: // Ouroboros (circle with arrowhead)
                ctx.arc(0, 0, s * 0.7, PI * 0.1, PI * 1.95);
                ctx.stroke();
                // Arrow head at mouth
                const hx = Math.cos(PI * 0.1) * s * 0.7;
                const hy = Math.sin(PI * 0.1) * s * 0.7;
                ctx.beginPath();
                ctx.moveTo(hx + s * 0.15, hy - s * 0.1);
                ctx.lineTo(hx, hy);
                ctx.lineTo(hx + s * 0.15, hy + s * 0.12);
                ctx.stroke();
                break;
            case 12: // Eye of Providence
                ctx.moveTo(-s, 0);
                ctx.quadraticCurveTo(0, -s * 0.8, s, 0);
                ctx.quadraticCurveTo(0, s * 0.8, -s, 0);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(0, 0, s * 0.25, 0, PI * 2);
                ctx.stroke();
                break;
            case 13: // Angular rune (Algiz-like)
                ctx.moveTo(0, s);
                ctx.lineTo(0, -s * 0.3);
                ctx.lineTo(-s * 0.6, -s);
                ctx.moveTo(0, -s * 0.3);
                ctx.lineTo(s * 0.6, -s);
                ctx.moveTo(0, s * 0.2);
                ctx.lineTo(-s * 0.4, -s * 0.3);
                ctx.moveTo(0, s * 0.2);
                ctx.lineTo(s * 0.4, -s * 0.3);
                ctx.stroke();
                break;
            case 14: // Infinity / Lemniscate
                ctx.moveTo(0, 0);
                ctx.bezierCurveTo(s * 0.5, -s * 0.7, s * 1.1, -s * 0.3, s * 0.6, 0);
                ctx.bezierCurveTo(s * 0.2, s * 0.3, -s * 0.2, s * 0.7, -s * 0.6, 0);
                ctx.bezierCurveTo(-s * 1.1, -s * 0.6, -s * 0.5, -s * 0.3, 0, 0);
                ctx.stroke();
                break;
            case 15: // Sulfur / Leviathan cross
            default:
                // Cross with infinity loop at bottom
                ctx.moveTo(0, -s);
                ctx.lineTo(0, s * 0.3);
                ctx.moveTo(-s * 0.45, -s * 0.3);
                ctx.lineTo(s * 0.45, -s * 0.3);
                ctx.stroke();
                // Triangle at bottom
                ctx.beginPath();
                ctx.moveTo(0, -s * 0.5);
                ctx.lineTo(-s * 0.4, s * 0.15);
                ctx.lineTo(s * 0.4, s * 0.15);
                ctx.closePath();
                ctx.stroke();
                // Infinity at base
                ctx.beginPath();
                ctx.arc(-s * 0.22, s * 0.65, s * 0.22, 0, PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(s * 0.22, s * 0.65, s * 0.22, 0, PI * 2);
                ctx.stroke();
                break;
        }
    }

    /**
     * Mycelial style — "Mushroom Colony"
     * Clusters of bioluminescent mushrooms with glowing caps, gill detail,
     * connecting mycelium threads, spore clouds, organic pulsing growth
     */
    renderMycelialStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const mirrors = config.mirrors;
        const rot = this.accumulatedRotation;
        const orbitFactor = config.orbitRadius / 200;

        const rotDir = this.seededRandom(seed) > 0.5 ? 1 : -1;
        const hueShift = this.seededRandom(seed + 2) * 40;

        // Organic palette
        const baseHue = hue;
        const tipHue = (hue + 120) % 360;
        const nodeHue = (hue + 260) % 360;
        const pulseHue = (hue + 30) % 360;

        // Helper: evaluate point on quadratic bezier at t
        const bezierAt = (x0, y0, cx, cy, x1, y1, t) => {
            const u = 1 - t;
            return [u * u * x0 + 2 * u * t * cx + t * t * x1, u * u * y0 + 2 * u * t * cy + t * t * y1];
        };

        // Primary branch count based on numSides
        const primaryCount = 3 + Math.min(2, Math.floor(numSides / 4));

        for (let m = 0; m < mirrors; m++) {
            const mirrorAngle = (Math.PI * 2 * m) / mirrors + rot * 0.15 * rotDir;

            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(mirrorAngle);

            // --- Build branch network as parallel arrays ---
            const bStartX = [], bStartY = [], bEndX = [], bEndY = [];
            const bCtrlX = [], bCtrlY = [], bGen = [], bParent = [];
            let branchCount = 0;

            // Primary branches: radiate outward from near center
            for (let p = 0; p < primaryCount; p++) {
                const pSeed = seed + 100 + m * 60 + p * 19;
                const spreadAngle = ((p / primaryCount) - 0.5) * (Math.PI / mirrors) * 0.85;
                const jitter = (this.seededRandom(pSeed) - 0.5) * 0.15;
                const angle = spreadAngle + jitter;
                const startDist = radius * 0.03;
                const len = radius * orbitFactor * (0.5 + this.seededRandom(pSeed + 1) * 0.3);

                const sx = Math.cos(angle) * startDist;
                const sy = Math.sin(angle) * startDist;
                const ex = Math.cos(angle) * (startDist + len);
                const ey = Math.sin(angle) * (startDist + len);

                // Perpendicular offset for organic curve, breathing with harmonic
                const perpX = -Math.sin(angle);
                const perpY = Math.cos(angle);
                const curveAmt = len * (0.1 + this.seededRandom(pSeed + 2) * 0.15) * (1 + harmonic * 0.3);
                const curveDir = this.seededRandom(pSeed + 3) > 0.5 ? 1 : -1;
                const cx = (sx + ex) / 2 + perpX * curveAmt * curveDir;
                const cy = (sy + ey) / 2 + perpY * curveAmt * curveDir;

                const idx = branchCount++;
                bStartX[idx] = sx; bStartY[idx] = sy;
                bEndX[idx] = ex; bEndY[idx] = ey;
                bCtrlX[idx] = cx; bCtrlY[idx] = cy;
                bGen[idx] = 0; bParent[idx] = -1;

                // Secondary branches (2-3 per primary)
                const secCount = 2 + (this.seededRandom(pSeed + 4) > 0.5 ? 1 : 0);
                for (let s = 0; s < secCount; s++) {
                    const sSeed = pSeed + 10 + s * 11;
                    const forkT = 0.4 + this.seededRandom(sSeed) * 0.3;
                    const [forkX, forkY] = bezierAt(sx, sy, cx, cy, ex, ey, forkT);
                    const forkAngle = Math.atan2(ey - sy, ex - sx) + (this.seededRandom(sSeed + 1) > 0.5 ? 1 : -1) * (0.4 + this.seededRandom(sSeed + 2) * 0.35);
                    const secLen = len * (0.5 + this.seededRandom(sSeed + 3) * 0.15);
                    const sex = forkX + Math.cos(forkAngle) * secLen;
                    const sey = forkY + Math.sin(forkAngle) * secLen;

                    const sPerpX = -Math.sin(forkAngle);
                    const sPerpY = Math.cos(forkAngle);
                    const sCurve = secLen * (0.08 + this.seededRandom(sSeed + 4) * 0.12) * (1 + harmonic * 0.3);
                    const sCurveDir = this.seededRandom(sSeed + 5) > 0.5 ? 1 : -1;
                    const scx = (forkX + sex) / 2 + sPerpX * sCurve * sCurveDir;
                    const scy = (forkY + sey) / 2 + sPerpY * sCurve * sCurveDir;

                    const sIdx = branchCount++;
                    bStartX[sIdx] = forkX; bStartY[sIdx] = forkY;
                    bEndX[sIdx] = sex; bEndY[sIdx] = sey;
                    bCtrlX[sIdx] = scx; bCtrlY[sIdx] = scy;
                    bGen[sIdx] = 1; bParent[sIdx] = idx;

                    // Tertiary branches (1-2 per secondary)
                    const terCount = 1 + (this.seededRandom(sSeed + 6) > 0.5 ? 1 : 0);
                    for (let t = 0; t < terCount; t++) {
                        const tSeed = sSeed + 20 + t * 7;
                        const tForkT = 0.5 + this.seededRandom(tSeed) * 0.3;
                        const [tForkX, tForkY] = bezierAt(forkX, forkY, scx, scy, sex, sey, tForkT);
                        const tAngle = Math.atan2(sey - forkY, sex - forkX) + (this.seededRandom(tSeed + 1) > 0.5 ? 1 : -1) * (0.3 + this.seededRandom(tSeed + 2) * 0.4);
                        const tLen = secLen * (0.3 + this.seededRandom(tSeed + 3) * 0.2);
                        const tex = tForkX + Math.cos(tAngle) * tLen;
                        const tey = tForkY + Math.sin(tAngle) * tLen;
                        const tPerpX = -Math.sin(tAngle);
                        const tPerpY = Math.cos(tAngle);
                        const tCurve = tLen * 0.1 * (1 + harmonic * 0.3);
                        const tcx = (tForkX + tex) / 2 + tPerpX * tCurve * (this.seededRandom(tSeed + 4) > 0.5 ? 1 : -1);
                        const tcy = (tForkY + tey) / 2 + tPerpY * tCurve * (this.seededRandom(tSeed + 4) > 0.5 ? 1 : -1);

                        const tIdx = branchCount++;
                        bStartX[tIdx] = tForkX; bStartY[tIdx] = tForkY;
                        bEndX[tIdx] = tex; bEndY[tIdx] = tey;
                        bCtrlX[tIdx] = tcx; bCtrlY[tIdx] = tcy;
                        bGen[tIdx] = 2; bParent[tIdx] = sIdx;
                    }
                }
            }

            // --- 1. Draw branch network (back layer) ---
            ctx.lineCap = 'round';
            for (let b = 0; b < branchCount; b++) {
                const gen = bGen[b];
                const branchHue = (baseHue + this.seededRandom(seed + b * 3) * 20 - 10 + hueShift) % 360;
                const widthMult = 1.2 - gen * 0.35;
                ctx.beginPath();
                ctx.moveTo(bStartX[b], bStartY[b]);
                ctx.quadraticCurveTo(bCtrlX[b], bCtrlY[b], bEndX[b], bEndY[b]);
                ctx.strokeStyle = `hsla(${branchHue}, ${config.saturation * 0.4}%, ${30 + energy * 20}%, ${0.3 + energy * 0.3})`;
                ctx.lineWidth = thickness * widthMult;
                ctx.stroke();
            }

            // --- 2. Nutrient pulses (traveling dots along branches) ---
            for (let b = 0; b < branchCount; b++) {
                const bSeed = seed + 700 + m * 50 + b * 3;
                const pulseSpeed = 0.3 + harmonic * 0.2;
                const pulseT = ((rot * pulseSpeed + this.seededRandom(bSeed) * 6.28) % 1.0 + 1.0) % 1.0;
                const [px, py] = bezierAt(bStartX[b], bStartY[b], bCtrlX[b], bCtrlY[b], bEndX[b], bEndY[b], pulseT);
                const dotR = thickness * 2 * (1 + energy * 0.5);

                const pGrad = ctx.createRadialGradient(px, py, 0, px, py, dotR * 2);
                pGrad.addColorStop(0, `hsla(${(pulseHue + hueShift) % 360}, ${config.saturation * 0.8}%, ${65 + energy * 15}%, ${0.4 + energy * 0.3})`);
                pGrad.addColorStop(1, `hsla(${(pulseHue + hueShift) % 360}, ${config.saturation * 0.5}%, 45%, 0)`);
                ctx.beginPath();
                ctx.arc(px, py, dotR * 2, 0, Math.PI * 2);
                ctx.fillStyle = pGrad;
                ctx.fill();
            }

            // --- 3. Growing tips (glow dots at tertiary endpoints) ---
            for (let b = 0; b < branchCount; b++) {
                if (bGen[b] !== 2) continue;
                const tSeed = seed + 900 + b * 5;
                const tipPulse = 1.5 + Math.sin(rot * 3 + this.seededRandom(tSeed) * 6.28) * 0.5;
                const tipR = thickness * tipPulse;
                const tipAlpha = 0.3 + energy * 0.4;

                const tGrad = ctx.createRadialGradient(bEndX[b], bEndY[b], 0, bEndX[b], bEndY[b], tipR * 2.5);
                tGrad.addColorStop(0, `hsla(${(tipHue + hueShift) % 360}, ${config.saturation * 0.8}%, ${70 + brightness * 10}%, ${tipAlpha})`);
                tGrad.addColorStop(1, `hsla(${(tipHue + hueShift) % 360}, ${config.saturation * 0.5}%, 50%, 0)`);
                ctx.beginPath();
                ctx.arc(bEndX[b], bEndY[b], tipR * 2.5, 0, Math.PI * 2);
                ctx.fillStyle = tGrad;
                ctx.fill();
            }

            // --- 4. Network nodes (luminous circles at fork junctions) ---
            for (let b = 0; b < branchCount; b++) {
                if (bGen[b] === 0) continue; // Only at forks (secondary/tertiary start = parent fork)
                const nodeR = thickness * 0.8 * (1 + harmonic * 0.3);
                const nodeAlpha = 0.2 + harmonic * 0.4;
                ctx.beginPath();
                ctx.arc(bStartX[b], bStartY[b], nodeR, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${(nodeHue + hueShift) % 360}, ${config.saturation * 0.6}%, ${50 + energy * 15}%, ${nodeAlpha})`;
                ctx.fill();
            }

            // --- 5. Spore drift (tiny floating particles) ---
            const sporeCount = 8 + Math.floor(brightness * 4);
            for (let s = 0; s < sporeCount; s++) {
                const sSeed = seed + 1200 + m * 20 + s * 7;
                const baseDist = radius * (0.1 + this.seededRandom(sSeed) * 0.6) * orbitFactor;
                const baseAngle = (this.seededRandom(sSeed + 1) - 0.5) * (Math.PI / mirrors) * 0.85;
                const driftX = Math.sin(rot * (0.5 + this.seededRandom(sSeed + 2) * 0.5) + this.seededRandom(sSeed + 3) * 6.28) * radius * 0.02;
                const driftY = Math.cos(rot * (0.4 + this.seededRandom(sSeed + 4) * 0.4) + this.seededRandom(sSeed + 5) * 6.28) * radius * 0.02;
                const sx = Math.cos(baseAngle) * baseDist + driftX;
                const sy = Math.sin(baseAngle) * baseDist + driftY;
                const sporeR = 1 + this.seededRandom(sSeed + 6);

                ctx.beginPath();
                ctx.arc(sx, sy, sporeR, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${(baseHue + hueShift) % 360}, ${config.saturation * 0.4}%, 60%, ${0.1 + brightness * 0.2})`;
                ctx.fill();
            }

            ctx.restore();
        }

        // --- 6. Central nexus (after mirror loop) ---
        const nexusR = radius * 0.06 * (1 + energy * 0.3);
        const nexusGrad = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, nexusR);
        nexusGrad.addColorStop(0, `hsla(${(baseHue + hueShift) % 360}, ${config.saturation * 0.7}%, ${60 + energy * 15}%, ${0.3 + energy * 0.35})`);
        nexusGrad.addColorStop(0.5, `hsla(${(nodeHue + hueShift) % 360}, ${config.saturation * 0.5}%, 45%, ${0.15 + energy * 0.15})`);
        nexusGrad.addColorStop(1, `hsla(${(baseHue + hueShift) % 360}, ${config.saturation * 0.3}%, 30%, 0)`);
        ctx.beginPath();
        ctx.arc(centerX, centerY, nexusR, 0, Math.PI * 2);
        ctx.fillStyle = nexusGrad;
        ctx.fill();
    }

    /**
     * Fluid background — 3-layer dark ferrofluid parallax (magnetic field rings, iron filings, threat pulses)
     */
    renderFluidBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const sat = config.saturation;

        // --- FAR LAYER (0.15x): Dark Magnetic Field Rings ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.15);

        for (let i = 0; i < 8; i++) {
            const ringRadius = maxDim * (0.1 + i * 0.11);
            const wobbleAmp = maxDim * 0.012 * (1 + energy * reactivity * 2 + i * 0.2);
            const segments = 48;
            const ringAlpha = 0.03 + reactivity * 0.03 + energy * reactivity * 0.02;

            ctx.beginPath();
            for (let s = 0; s <= segments; s++) {
                const angle = (Math.PI * 2 * s) / segments;
                const wobble = Math.sin(angle * 4 + rot * 0.6 + i * 0.7) * wobbleAmp
                    + Math.sin(angle * 9 + rot * 1.1 - i * 0.3) * wobbleAmp * 0.4;
                const r = ringRadius + wobble;
                const x = Math.cos(angle) * r;
                const y = Math.sin(angle) * r;
                if (s === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.strokeStyle = `hsla(${bgHue}, ${sat * 0.2}%, ${20 + i * 1.5}%, ${ringAlpha * (1 - i * 0.08)})`;
            ctx.lineWidth = 0.5 + energy * reactivity * 0.5;
            ctx.stroke();
        }
        ctx.restore();

        // --- MID LAYER (-0.5x): Iron Filings ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.5);

        for (let i = 0; i < 20; i++) {
            const seed = i * 37 + 89;
            const dist = maxDim * (0.08 + this.seededRandom(seed) * 0.85);
            const baseAngle = this.seededRandom(seed + 1) * Math.PI * 2;
            const orbitSpeed = 0.1 + this.seededRandom(seed + 2) * 0.2;
            const angle = baseAngle + rot * orbitSpeed;
            const size = 1 + this.seededRandom(seed + 3) * 2;
            const alpha = 0.03 + reactivity * 0.05;

            const dx = Math.cos(angle) * dist;
            const dy = Math.sin(angle) * dist;

            ctx.beginPath();
            ctx.arc(dx, dy, size, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${bgHue}, ${sat * 0.15}%, 15%, ${alpha})`;
            ctx.fill();
        }
        ctx.restore();

        // --- NEAR LAYER (1.0x): Threat Pulses ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 1.0);

        for (let i = 0; i < 6; i++) {
            const seed = i * 23 + 197;
            const dist = maxDim * (0.15 + this.seededRandom(seed) * 0.6);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const phase = this.seededRandom(seed + 2) * Math.PI * 2;
            const pulse = 0.5 + Math.sin(rot * 3 + phase) * 0.5;
            const glowSize = maxDim * (0.02 + this.seededRandom(seed + 3) * 0.03) * (0.8 + energy * reactivity * 0.6);
            const alpha = (0.04 + reactivity * 0.08) * pulse;

            const dx = Math.cos(angle) * dist;
            const dy = Math.sin(angle) * dist;

            const glowGrad = ctx.createRadialGradient(dx, dy, 0, dx, dy, glowSize);
            glowGrad.addColorStop(0, `hsla(${bgHue}, ${sat * 0.8}%, 50%, ${alpha})`);
            glowGrad.addColorStop(0.5, `hsla(${bgHue}, ${sat * 0.6}%, 35%, ${alpha * 0.4})`);
            glowGrad.addColorStop(1, `hsla(${bgHue}, ${sat * 0.3}%, 20%, 0)`);
            ctx.beginPath();
            ctx.arc(dx, dy, glowSize, 0, Math.PI * 2);
            ctx.fillStyle = glowGrad;
            ctx.fill();
        }
        ctx.restore();
    }

    /**
     * Fluid style — Dark ferrofluid / Rezz-inspired. Sharp magnetic spikes, dark mass, menacing energy.
     */
    renderFluidStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const mirrors = config.mirrors;
        const rot = this.accumulatedRotation;
        const orbitFactor = config.orbitRadius / 200;
        const sat = config.saturation;

        // Seed-based variation
        const rotDir = this.seededRandom(seed) > 0.5 ? 1 : -1;
        const wobbleFreq = 1.5 + this.seededRandom(seed + 2) * 1.5;

        // --- Mirror segments ---
        for (let m = 0; m < mirrors; m++) {
            const mirrorAngle = (Math.PI * 2 * m) / mirrors + rot * 0.15 * rotDir;

            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(mirrorAngle);

            // 1. Magnetic field lines — concentric distorted rings
            const fieldRingCount = 3 + Math.floor(energy);
            for (let i = 0; i < fieldRingCount; i++) {
                const ringRadius = radius * (0.15 + i * 0.15) * orbitFactor;
                const segments = 48;
                const distortAmp = ringRadius * 0.06 * (1 + energy * 2);
                const ringAlpha = 0.15 + energy * 0.2;

                ctx.beginPath();
                for (let s = 0; s <= segments; s++) {
                    const angle = (Math.PI * 2 * s) / segments;
                    const distort = Math.sin(angle * 5 + rot * 0.8 + i * 1.2) * distortAmp
                        + Math.sin(angle * 11 + rot * 1.5 - i) * distortAmp * 0.3;
                    const r = ringRadius + distort;
                    const x = Math.cos(angle) * r;
                    const y = Math.sin(angle) * r;
                    if (s === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.closePath();
                ctx.strokeStyle = `hsla(${hue}, ${sat * 0.3}%, ${20 + i * 2}%, ${ringAlpha * (1 - i * 0.15)})`;
                ctx.lineWidth = thickness * 0.3;
                ctx.stroke();
            }

            // 2. Ferrofluid spike crown — the core visual
            const spikeCount = 6 + numSides + Math.floor(energy * 4);
            const segAngle = Math.PI / mirrors;

            for (let s = 0; s < spikeCount; s++) {
                const sSeed = seed + 100 + m * 50 + s * 13;
                const baseAngle = (s / spikeCount) * segAngle;
                const sway = Math.sin(rot * 1.5 + this.seededRandom(sSeed) * Math.PI * 2) * 0.08;
                const spikeAngle = baseAngle + sway;
                const seeded = this.seededRandom(sSeed + 1);
                const spikeLen = radius * (0.08 + energy * 0.25 + seeded * 0.12) * orbitFactor;
                const spikeBaseWidth = radius * 0.018 * (0.6 + energy * 0.4);
                const baseDist = radius * 0.14 * orbitFactor;

                // Base and tip positions
                const bx = Math.cos(spikeAngle) * baseDist;
                const by = Math.sin(spikeAngle) * baseDist;
                const tipX = Math.cos(spikeAngle) * (baseDist + spikeLen);
                const tipY = Math.sin(spikeAngle) * (baseDist + spikeLen);
                const perpAngle = spikeAngle + Math.PI / 2;

                // Tapered spike with quadratic curve for organic shape
                const leftBX = bx + Math.cos(perpAngle) * spikeBaseWidth;
                const leftBY = by + Math.sin(perpAngle) * spikeBaseWidth;
                const rightBX = bx - Math.cos(perpAngle) * spikeBaseWidth;
                const rightBY = by - Math.sin(perpAngle) * spikeBaseWidth;
                const midX = (bx + tipX) / 2 + Math.cos(perpAngle) * spikeBaseWidth * 0.3;
                const midY = (by + tipY) / 2 + Math.sin(perpAngle) * spikeBaseWidth * 0.3;
                const midX2 = (bx + tipX) / 2 - Math.cos(perpAngle) * spikeBaseWidth * 0.3;
                const midY2 = (by + tipY) / 2 - Math.sin(perpAngle) * spikeBaseWidth * 0.3;

                ctx.beginPath();
                ctx.moveTo(leftBX, leftBY);
                ctx.quadraticCurveTo(midX, midY, tipX, tipY);
                ctx.quadraticCurveTo(midX2, midY2, rightBX, rightBY);
                ctx.closePath();

                // Gradient: near-black body → accent-colored glowing tip
                const spikeGrad = ctx.createLinearGradient(bx, by, tipX, tipY);
                const bodyAlpha = 0.5 + energy * 0.3;
                const tipGlow = 55 + energy * 20;
                spikeGrad.addColorStop(0, `hsla(${hue}, ${sat * 0.2}%, 12%, ${bodyAlpha})`);
                spikeGrad.addColorStop(0.7, `hsla(${hue}, ${sat * 0.3}%, 15%, ${bodyAlpha * 0.8})`);
                spikeGrad.addColorStop(0.85, `hsla(${hue}, ${sat * 0.7}%, ${tipGlow * 0.6}%, ${bodyAlpha * 0.6})`);
                spikeGrad.addColorStop(1, `hsla(${hue}, ${sat * 0.9}%, ${tipGlow}%, ${bodyAlpha * 0.9})`);
                ctx.fillStyle = spikeGrad;
                ctx.fill();
            }

            // 3. Menacing core mass — dark undulating blob
            const massRadius = radius * (0.14 + harmonic * 0.04) * orbitFactor;
            const wobX = Math.sin(rot * wobbleFreq) * radius * 0.01;
            const wobY = Math.cos(rot * wobbleFreq * 0.7) * radius * 0.008;
            const massAlpha = 0.5 + energy * 0.3;

            const massGrad = ctx.createRadialGradient(
                wobX, wobY, 0,
                wobX, wobY, massRadius
            );
            massGrad.addColorStop(0, `hsla(${hue}, ${sat * 0.15}%, 8%, ${massAlpha})`);
            massGrad.addColorStop(0.6, `hsla(${hue}, ${sat * 0.2}%, 12%, ${massAlpha * 0.9})`);
            massGrad.addColorStop(1, `hsla(${hue}, ${sat * 0.15}%, 10%, ${massAlpha * 0.3})`);

            ctx.beginPath();
            ctx.arc(wobX, wobY, massRadius, 0, Math.PI * 2);
            ctx.fillStyle = massGrad;
            ctx.fill();

            // 4. Drip tendrils — thin dark bezier curves from mass edge
            const dripCount = 3 + Math.floor(harmonic * 2);
            for (let d = 0; d < dripCount; d++) {
                const dSeed = seed + 400 + m * 30 + d * 11;
                const dripAngle = this.seededRandom(dSeed) * segAngle;
                const dripLen = radius * (0.08 + harmonic * 0.12) * orbitFactor * (0.5 + this.seededRandom(dSeed + 1) * 0.5);
                const startR = massRadius * 0.9;
                const sx = Math.cos(dripAngle) * startR + wobX;
                const sy = Math.sin(dripAngle) * startR + wobY;
                const ex = Math.cos(dripAngle + (this.seededRandom(dSeed + 2) - 0.5) * 0.5) * (startR + dripLen);
                const ey = Math.sin(dripAngle + (this.seededRandom(dSeed + 2) - 0.5) * 0.5) * (startR + dripLen);
                const cpx = (sx + ex) / 2 + (this.seededRandom(dSeed + 3) - 0.5) * dripLen * 0.6;
                const cpy = (sy + ey) / 2 + (this.seededRandom(dSeed + 4) - 0.5) * dripLen * 0.6;

                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.quadraticCurveTo(cpx, cpy, ex, ey);
                ctx.strokeStyle = `hsla(${hue}, ${sat * 0.2}%, 15%, ${0.2 + harmonic * 0.2})`;
                ctx.lineWidth = thickness * 0.4;
                ctx.lineCap = 'round';
                ctx.stroke();

                // Bright droplet at terminus
                ctx.beginPath();
                ctx.arc(ex, ey, 1.5, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${hue}, ${sat * 0.8}%, ${50 + brightness * 20}%, ${0.3 + brightness * 0.4})`;
                ctx.fill();
            }

            // 5. Magnetic dust — tiny oriented particles near spikes
            const dustCount = 10 + Math.floor(brightness * 5);
            for (let p = 0; p < dustCount; p++) {
                const pSeed = seed + 600 + m * 40 + p * 7;
                const pAngle = this.seededRandom(pSeed) * segAngle;
                const pDist = radius * (0.1 + this.seededRandom(pSeed + 1) * 0.35) * orbitFactor;
                const px = Math.cos(pAngle) * pDist;
                const py = Math.sin(pAngle) * pDist;
                const pSize = 1 + this.seededRandom(pSeed + 2);
                const pAlpha = 0.2 + brightness * 0.3;

                ctx.beginPath();
                ctx.arc(px, py, pSize, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${hue}, ${sat * 0.3}%, 20%, ${pAlpha})`;
                ctx.fill();
            }

            ctx.restore();
        }

        // --- Central vortex (after mirror loop) — dark core with bright rim ---
        const vortexRadius = radius * 0.08 * (1 + energy * 0.4);
        const vortexGrad = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, vortexRadius
        );
        vortexGrad.addColorStop(0, `hsla(0, 0%, 5%, 0.7)`);
        vortexGrad.addColorStop(0.7, `hsla(0, 0%, 3%, 0.5)`);
        vortexGrad.addColorStop(0.88, `hsla(${hue}, ${sat * 0.8}%, ${45 + energy * 15}%, ${0.4 + energy * 0.3})`);
        vortexGrad.addColorStop(1, `hsla(${hue}, ${sat * 0.6}%, 30%, 0)`);

        ctx.beginPath();
        ctx.arc(centerX, centerY, vortexRadius, 0, Math.PI * 2);
        ctx.fillStyle = vortexGrad;
        ctx.fill();

        // Bright rim stroke — "the eye"
        ctx.beginPath();
        ctx.arc(centerX, centerY, vortexRadius * 0.9, 0, Math.PI * 2);
        ctx.strokeStyle = `hsla(${hue}, ${sat * 0.8}%, 50%, ${0.2 + energy * 0.3})`;
        ctx.lineWidth = thickness * 0.4;
        ctx.stroke();
    }

    /**
     * Orrery background — Counter-rotating star chart with etched orbital trails
     * 3 parallax layers: star field (far), orbital trails (mid), brass dust (near)
     */
    renderOrreryBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const sat = config.saturation;

        // --- FAR LAYER (−0.25x): Counter-rotating star chart ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.25);

        // Star field — scattered fixed stars
        const starCount = 40;
        for (let i = 0; i < starCount; i++) {
            const seed = i * 41 + 17;
            const dist = maxDim * (0.1 + this.seededRandom(seed) * 0.95);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const starSize = 0.5 + this.seededRandom(seed + 2) * 1.5;
            const twinkle = 0.5 + Math.sin(rot * 4 + this.seededRandom(seed + 3) * Math.PI * 2) * 0.5;
            const alpha = (0.04 + reactivity * 0.06) * twinkle;

            const sx = Math.cos(angle) * dist;
            const sy = Math.sin(angle) * dist;

            ctx.beginPath();
            ctx.arc(sx, sy, starSize, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${(bgHue + 30 + this.seededRandom(seed + 4) * 40) % 360}, ${sat * 0.3}%, ${70 + this.seededRandom(seed + 5) * 20}%, ${alpha})`;
            ctx.fill();
        }

        // Constellation lines — faint connecting lines between nearby stars
        for (let i = 0; i < 12; i++) {
            const seed1 = i * 41 + 17;
            const seed2 = ((i + 1 + Math.floor(this.seededRandom(seed1 + 10) * 3)) % starCount) * 41 + 17;
            const d1 = maxDim * (0.1 + this.seededRandom(seed1) * 0.95);
            const a1 = this.seededRandom(seed1 + 1) * Math.PI * 2;
            const d2 = maxDim * (0.1 + this.seededRandom(seed2) * 0.95);
            const a2 = this.seededRandom(seed2 + 1) * Math.PI * 2;

            ctx.beginPath();
            ctx.moveTo(Math.cos(a1) * d1, Math.sin(a1) * d1);
            ctx.lineTo(Math.cos(a2) * d2, Math.sin(a2) * d2);
            ctx.strokeStyle = `hsla(${bgHue}, ${sat * 0.15}%, 50%, ${0.02 + reactivity * 0.03})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }
        ctx.restore();

        // --- MID LAYER (0.6x): Phosphor orbital trails ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.6);

        const trailCount = 5;
        for (let i = 0; i < trailCount; i++) {
            const orbitRadius = maxDim * (0.15 + i * 0.14);
            const eccentricity = 0.15 + this.seededRandom(i * 31 + 7) * 0.25;
            const trailHue = (bgHue + i * 25 + harmonic * 20) % 360;
            const segments = 80;
            const alpha = 0.03 + reactivity * 0.05 + energy * reactivity * 0.03;

            ctx.beginPath();
            for (let s = 0; s <= segments; s++) {
                const angle = (Math.PI * 2 * s) / segments;
                const r = orbitRadius * (1 - eccentricity * Math.cos(angle));
                const x = Math.cos(angle) * r;
                const y = Math.sin(angle) * r;
                s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.strokeStyle = `hsla(${trailHue}, ${sat * 0.4}%, 55%, ${alpha})`;
            ctx.lineWidth = 0.6 + energy * reactivity * 0.8;
            ctx.stroke();
        }
        ctx.restore();

        // --- NEAR LAYER (1.3x): Floating brass dust motes ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 1.3);

        const dustCount = 16;
        for (let i = 0; i < dustCount; i++) {
            const seed = i * 29 + 113;
            const dist = maxDim * (0.1 + this.seededRandom(seed) * 0.8);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const phase = this.seededRandom(seed + 2) * Math.PI * 2;
            const drift = Math.sin(rot * 2 + phase) * maxDim * 0.02;
            const glowSize = 2 + this.seededRandom(seed + 3) * 3 + energy * reactivity * 2;
            const alpha = 0.04 + reactivity * 0.06;
            const dustHue = (bgHue + 20 + this.seededRandom(seed + 4) * 30) % 360;

            const dx = Math.cos(angle) * (dist + drift);
            const dy = Math.sin(angle) * (dist + drift);

            const glowGrad = ctx.createRadialGradient(dx, dy, 0, dx, dy, glowSize);
            glowGrad.addColorStop(0, `hsla(${dustHue}, ${sat * 0.6}%, 65%, ${alpha})`);
            glowGrad.addColorStop(1, `hsla(${dustHue}, ${sat * 0.3}%, 40%, 0)`);
            ctx.beginPath();
            ctx.arc(dx, dy, glowSize, 0, Math.PI * 2);
            ctx.fillStyle = glowGrad;
            ctx.fill();
        }
        ctx.restore();
    }

    /**
     * Update binary star orrery simulation with N-body planet interactions.
     * Binary stars are computed analytically (guaranteed stable circular orbit).
     * Planets experience gravity from both stars and all other planets.
     *
     * @param {number} dt - Time step
     * @param {number} bassEnergy - Current bass/percussive energy [0,1]
     * @param {number} seed - Shape seed for deterministic initialization
     * @returns {{ starA: {x,y}, starB: {x,y}, binaryAngle: number, planets: Array }}
     */
    _updateBinaryOrrery(dt, bassEnergy, seed) {
        const G = 1.0;
        const massA = 1.0;
        const massB = 0.8;
        const totalMass = massA + massB;
        const binarySep = 0.5; // semi-major axis of binary

        // --- Planet specification table ---
        const planetSpecs = [
            { name: "vulcan",   R: 1.3, mass: 0.0005 },
            { name: "terra",    R: 1.7, mass: 0.003  },
            { name: "aridus",   R: 2.1, mass: 0.001  },
            { name: "colossus", R: 2.9, mass: 0.05   },
            { name: "aurelian", R: 3.6, mass: 0.03   },
            { name: "glacius",  R: 4.2, mass: 0.008  },
            { name: "tempest",  R: 4.8, mass: 0.006  },
            { name: "wanderer", R: 5.5, mass: 0.0001 },
        ];

        // --- Initialize or reset state on seed change ---
        if (!this._binaryState || this._binarySeed !== seed) {
            this._binarySeed = seed;
            this._binaryLastRot = null;

            const planets = planetSpecs.map((spec, i) => {
                const phase = this.seededRandom(seed + i * 137 + 7) * Math.PI * 2;
                const circularV = Math.sqrt(G * totalMass / spec.R);
                // Slight eccentricity perturbation from seed (±4%)
                const perturbation = 1.0 + (this.seededRandom(seed + i * 251 + 41) - 0.5) * 0.08;
                const v = circularV * perturbation;
                return {
                    x: spec.R * Math.cos(phase),
                    y: spec.R * Math.sin(phase),
                    vx: -v * Math.sin(phase),
                    vy:  v * Math.cos(phase),
                    mass: spec.mass,
                    trail: [],
                };
            });

            this._binaryState = {
                binaryAngle: 0,
                planets: planets,
            };
        }

        const state = this._binaryState;

        // --- Binary angular velocity (analytical circular orbit) ---
        const omega = Math.sqrt(G * totalMass / (binarySep * binarySep * binarySep));

        // Bass energy speeds up binary rotation
        const bassSpeedup = 1.0 + bassEnergy * 2.0;
        state.binaryAngle += omega * dt * bassSpeedup;

        // --- Analytical star positions (orbit barycenter at origin) ---
        // Star A orbits at distance (massB/totalMass)*a from barycenter
        const rA = (massB / totalMass) * binarySep;
        const rB = (massA / totalMass) * binarySep;
        const angle = state.binaryAngle;

        const starA = {
            x:  rA * Math.cos(angle),
            y:  rA * Math.sin(angle),
        };
        const starB = {
            x: -rB * Math.cos(angle),
            y: -rB * Math.sin(angle),
        };

        // --- N-body integration for planets (8 substeps) ---
        const substeps = 8;
        const subDt = dt / substeps;
        const planets = state.planets;

        for (let step = 0; step < substeps; step++) {
            // Compute accelerations for all planets
            const ax = new Float64Array(planets.length);
            const ay = new Float64Array(planets.length);

            for (let i = 0; i < planets.length; i++) {
                const p = planets[i];

                // Gravity from Star A
                let dxA = starA.x - p.x;
                let dyA = starA.y - p.y;
                let distA = Math.sqrt(dxA * dxA + dyA * dyA);
                distA = Math.max(distA, 0.05); // softening for stars
                let forceA = G * massA / (distA * distA * distA);
                ax[i] += forceA * dxA;
                ay[i] += forceA * dyA;

                // Gravity from Star B
                let dxB = starB.x - p.x;
                let dyB = starB.y - p.y;
                let distB = Math.sqrt(dxB * dxB + dyB * dyB);
                distB = Math.max(distB, 0.05);
                let forceB = G * massB / (distB * distB * distB);
                ax[i] += forceB * dxB;
                ay[i] += forceB * dyB;

                // Gravity from all other planets (N-body)
                for (let j = 0; j < planets.length; j++) {
                    if (j === i) continue;
                    const other = planets[j];
                    let dx = other.x - p.x;
                    let dy = other.y - p.y;
                    let dist = Math.sqrt(dx * dx + dy * dy);
                    let safeDist = Math.max(dist, 0.15); // softened for planet-planet
                    let force = G * other.mass / (safeDist * safeDist * safeDist);
                    ax[i] += force * dx;
                    ay[i] += force * dy;
                }
            }

            // Apply accelerations and integrate positions
            for (let i = 0; i < planets.length; i++) {
                const p = planets[i];

                // Bass reactivity: tangential kick to speed up orbits
                if (bassEnergy > 0.3) {
                    const r = Math.sqrt(p.x * p.x + p.y * p.y);
                    if (r > 0.01) {
                        const tangX = -p.y / r;
                        const tangY =  p.x / r;
                        const kick = (bassEnergy - 0.3) * 0.3 * subDt;
                        p.vx += tangX * kick;
                        p.vy += tangY * kick;
                    }
                }

                p.vx += ax[i] * subDt;
                p.vy += ay[i] * subDt;

                // Speed clamp
                let speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
                if (speed > 4.0) {
                    p.vx *= 4.0 / speed;
                    p.vy *= 4.0 / speed;
                }

                p.x += p.vx * subDt;
                p.y += p.vy * subDt;

                // Soft boundary at r > 6.0 — gentle inward nudge
                let r = Math.sqrt(p.x * p.x + p.y * p.y);
                if (r > 6.0) {
                    let pullback = (r - 6.0) * 0.5 * subDt;
                    p.vx -= (p.x / r) * pullback;
                    p.vy -= (p.y / r) * pullback;
                }
            }
        }

        // --- Record trails (last 80 positions per planet) ---
        for (let i = 0; i < planets.length; i++) {
            const p = planets[i];
            p.trail.push({ x: p.x, y: p.y });
            if (p.trail.length > 150) {
                p.trail.shift();
            }
        }

        return {
            starA: starA,
            starB: starB,
            binaryAngle: state.binaryAngle,
            planets: planets,
        };
    }

    /**
     * Render the Orrery visualization style — a mechanical brass orrery
     * with tilted perspective disc, concentric orbit tracks, armature arms,
     * and planets on stalks above the disc plane.
     */
    renderOrreryStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const rot = this.accumulatedRotation;
        const seed = config.shapeSeed;
        const sat = config.saturation || 80;
        const mirrors = config.mirrors;

        // ORBIT = zoom, SIZE = density (more planets/detail)
        const orbitFactor = (config.orbitRadius || 200) / 200;
        const sizeNorm = (config.baseRadius || 150) / 150;
        const density = sizeNorm;
        const visiblePlanets = Math.min(8, Math.max(2, Math.floor(1 + sizeNorm * 4.5)));

        // =====================================================================
        // Simulation
        // =====================================================================
        let simDt = 0;
        if (this._binaryLastRot !== null && this._binaryLastRot !== undefined) {
            let delta = rot - this._binaryLastRot;
            if (Math.abs(delta) > 0.5) delta = 0;
            simDt = Math.max(0.002, Math.min(Math.abs(delta), 0.01));
        }
        this._binaryLastRot = rot;

        const sim = this._updateBinaryOrrery(simDt, energy, seed);
        const { starA, starB, binaryAngle, planets } = sim;

        // =====================================================================
        // Orrery geometry
        // =====================================================================
        const canvasBase = Math.min(centerX, centerY);
        const discRadius = canvasBase * 0.75 * orbitFactor;
        const tilt = 0.38; // perspective compression
        const simScale = discRadius / 6.0; // map simulation radius 6 to disc edge

        // Planet rendering specs
        const planetSpecs = [
            { name: "vulcan",   sizeM: 0.015, color: [30, 15, 45],  ring: false },
            { name: "terra",    sizeM: 0.022, color: [200, 65, 50],  ring: false },
            { name: "aridus",   sizeM: 0.018, color: [15, 75, 48],   ring: false },
            { name: "colossus", sizeM: 0.065, color: [30, 60, 58],   ring: false },
            { name: "aurelian", sizeM: 0.050, color: [42, 55, 62],   ring: true  },
            { name: "glacius",  sizeM: 0.032, color: [178, 55, 60],  ring: true  },
            { name: "tempest",  sizeM: 0.028, color: [230, 60, 48],  ring: false },
            { name: "wanderer", sizeM: 0.010, color: [210, 20, 75],  ring: false },
        ];

        const brassHue = 42;
        const brassSat = 65;

        ctx.save();
        ctx.translate(centerX, centerY);

        // =====================================================================
        // 1. THE BRASS DISC — filled tilted ellipse with golden gradient
        // =====================================================================
        ctx.save();
        ctx.scale(1, tilt);

        // Outer disc shadow
        const discShadow = ctx.createRadialGradient(0, 0, discRadius * 0.5, 0, 0, discRadius * 1.05);
        discShadow.addColorStop(0, `hsla(${brassHue}, 40%, 8%, 0)`);
        discShadow.addColorStop(0.85, `hsla(${brassHue}, 40%, 5%, 0.3)`);
        discShadow.addColorStop(1, `hsla(${brassHue}, 40%, 3%, 0.6)`);
        ctx.beginPath();
        ctx.arc(0, 0, discRadius * 1.05, 0, Math.PI * 2);
        ctx.fillStyle = discShadow;
        ctx.fill();

        // Main disc body
        const discGrad = ctx.createRadialGradient(
            -discRadius * 0.2, -discRadius * 0.2, discRadius * 0.05,
            0, 0, discRadius
        );
        discGrad.addColorStop(0, `hsla(${brassHue}, ${brassSat}%, 45%, 0.25)`);
        discGrad.addColorStop(0.3, `hsla(${brassHue}, ${brassSat - 10}%, 30%, 0.20)`);
        discGrad.addColorStop(0.7, `hsla(${brassHue}, ${brassSat - 20}%, 20%, 0.18)`);
        discGrad.addColorStop(1, `hsla(${brassHue}, ${brassSat - 25}%, 12%, 0.15)`);
        ctx.beginPath();
        ctx.arc(0, 0, discRadius, 0, Math.PI * 2);
        ctx.fillStyle = discGrad;
        ctx.fill();

        // Disc rim
        ctx.beginPath();
        ctx.arc(0, 0, discRadius, 0, Math.PI * 2);
        ctx.strokeStyle = `hsla(${brassHue}, ${brassSat}%, 55%, 0.5)`;
        ctx.lineWidth = thickness * 0.6;
        ctx.stroke();

        // =====================================================================
        // 2. CONCENTRIC ORBIT TRACK RINGS
        // =====================================================================
        const orbitRadii = [1.3, 1.7, 2.1, 2.9, 3.6, 4.2, 4.8, 5.5];
        for (let i = 0; i < Math.min(visiblePlanets, orbitRadii.length); i++) {
            const trackR = orbitRadii[i] * simScale;
            ctx.beginPath();
            ctx.arc(0, 0, trackR, 0, Math.PI * 2);
            const isGiant = (i === 3 || i === 4);
            ctx.strokeStyle = `hsla(${brassHue}, ${brassSat}%, ${isGiant ? 50 : 40}%, ${isGiant ? 0.35 : 0.20})`;
            ctx.lineWidth = thickness * (isGiant ? 0.35 : 0.2);
            ctx.stroke();
        }

        // Inner decorative rings (zodiac-style) — count driven by mirrors
        const decoRings = Math.max(1, Math.floor(mirrors / 2));
        for (let r = 0; r < decoRings; r++) {
            const innerR = discRadius * (0.08 + r * (0.12 / decoRings));
            ctx.beginPath();
            ctx.arc(0, 0, innerR, 0, Math.PI * 2);
            ctx.strokeStyle = `hsla(${brassHue}, ${brassSat}%, 50%, 0.15)`;
            ctx.lineWidth = thickness * 0.15;
            ctx.stroke();
        }

        // Decorative tick marks on inner ring — count driven by mirrors
        const tickR = discRadius * 0.18;
        const tickCount = mirrors * 3;
        for (let t = 0; t < tickCount; t++) {
            const tAngle = (t / tickCount) * Math.PI * 2;
            const inner = tickR * 0.8;
            const outer = tickR;
            if (numSides <= 4) {
                // Simple line ticks for low numSides
                ctx.beginPath();
                ctx.moveTo(Math.cos(tAngle) * inner, Math.sin(tAngle) * inner);
                ctx.lineTo(Math.cos(tAngle) * outer, Math.sin(tAngle) * outer);
                ctx.strokeStyle = `hsla(${brassHue}, ${brassSat}%, 50%, 0.12)`;
                ctx.lineWidth = thickness * 0.15;
                ctx.stroke();
            } else {
                // Polygon marker ticks for higher numSides
                const mx = Math.cos(tAngle) * (inner + outer) * 0.5;
                const my = Math.sin(tAngle) * (inner + outer) * 0.5;
                const markerSize = (outer - inner) * 0.4;
                ctx.beginPath();
                for (let p = 0; p < numSides; p++) {
                    const pa = (Math.PI * 2 * p) / numSides + tAngle;
                    const px = mx + Math.cos(pa) * markerSize;
                    const py = my + Math.sin(pa) * markerSize;
                    p === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
                }
                ctx.closePath();
                ctx.fillStyle = `hsla(${brassHue}, ${brassSat}%, 50%, 0.10)`;
                ctx.fill();
                ctx.strokeStyle = `hsla(${brassHue}, ${brassSat}%, 50%, 0.12)`;
                ctx.lineWidth = thickness * 0.1;
                ctx.stroke();
            }
        }

        ctx.restore(); // end disc tilt

        // =====================================================================
        // 3. Build planet rendering list with screen positions, depth-sort
        // =====================================================================
        const renderList = [];

        // Binary stars → screen positions on disc plane
        const sAx = starA.x * simScale;
        const sAy = starA.y * simScale * tilt;
        const sBx = starB.x * simScale;
        const sBy = starB.y * simScale * tilt;

        // Planets
        for (let i = 0; i < Math.min(visiblePlanets, planets.length); i++) {
            const p = planets[i];
            const spec = planetSpecs[i];
            const discX = p.x * simScale;
            const discY = p.y * simScale * tilt; // position on disc surface
            const bodySize = canvasBase * spec.sizeM * orbitFactor;
            const stalkHeight = bodySize * 1.8 + 8; // how far above disc

            renderList.push({
                i, spec, p,
                discX, discY,          // where the stalk meets the disc
                planetX: discX,        // planet sphere X (same as disc)
                planetY: discY - stalkHeight, // planet sphere Y (above disc)
                bodySize,
                stalkHeight,
                depth: discY,          // sort by disc Y (further down = closer to viewer)
            });
        }

        // Sort back-to-front (smaller depth first = further away drawn first)
        renderList.sort((a, b) => a.depth - b.depth);

        // =====================================================================
        // 4. ARMATURE ARMS — brass lines from center to planet on disc
        // =====================================================================
        ctx.save();
        for (const item of renderList) {
            // Arm on disc surface
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(item.discX, item.discY);
            ctx.strokeStyle = `hsla(${brassHue}, ${brassSat}%, 45%, 0.25)`;
            ctx.lineWidth = thickness * 0.45;
            ctx.stroke();

            // Small brass joint at disc end
            ctx.beginPath();
            ctx.arc(item.discX, item.discY, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${brassHue}, ${brassSat}%, 55%, 0.5)`;
            ctx.fill();
        }
        ctx.restore();

        // =====================================================================
        // 5. PLASMA BRIDGE between stars (subtle on disc plane)
        // =====================================================================
        for (let strand = 0; strand < 2; strand++) {
            ctx.beginPath();
            const segs = 16;
            for (let s = 0; s <= segs; s++) {
                const t = s / segs;
                const baseX = sAx + (sBx - sAx) * t;
                const baseY = sAy + (sBy - sAy) * t;
                const perpX = -(sBy - sAy);
                const perpY = (sBx - sAx);
                const perpLen = Math.sqrt(perpX * perpX + perpY * perpY) || 1;
                const wave = Math.sin(t * Math.PI * 3 + rot * 12 + strand * 2.5) * (2 + energy * 5);
                const px = baseX + (perpX / perpLen) * wave * (strand - 0.5) * 0.8;
                const py = baseY + (perpY / perpLen) * wave * (strand - 0.5) * 0.8;
                if (s === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.strokeStyle = `hsla(${brassHue + 15}, 80%, 75%, ${0.06 + energy * 0.15})`;
            ctx.lineWidth = thickness * (0.25 + energy * 0.3);
            ctx.stroke();
        }

        // =====================================================================
        // 6. CENTRAL SUN — bright glow on the disc
        // =====================================================================
        const sunSize = canvasBase * 0.045 * orbitFactor * (1.0 + energy * 0.4);

        // Outer corona
        const corona = ctx.createRadialGradient(0, 0, sunSize * 0.2, 0, 0, sunSize * 4);
        corona.addColorStop(0, `hsla(${brassHue}, 90%, 80%, 0.6)`);
        corona.addColorStop(0.3, `hsla(${brassHue}, 80%, 60%, 0.2)`);
        corona.addColorStop(0.6, `hsla(${brassHue}, 70%, 40%, 0.05)`);
        corona.addColorStop(1, `hsla(${brassHue}, 60%, 30%, 0)`);
        ctx.beginPath();
        ctx.arc(0, 0, sunSize * 4, 0, Math.PI * 2);
        ctx.fillStyle = corona;
        ctx.fill();

        // Sun flare spikes
        const flareCount = Math.floor(4 + density * 6);
        if (energy > 0.15) {
            for (let f = 0; f < flareCount; f++) {
                const fAngle = (f / flareCount) * Math.PI * 2 + rot * 3;
                const flareLen = sunSize * (1.5 + energy * 3.0) * (0.6 + this.seededRandom(seed + f * 47) * 0.8);
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(Math.cos(fAngle) * flareLen, Math.sin(fAngle) * flareLen);
                ctx.strokeStyle = `hsla(${brassHue}, 100%, 85%, ${energy * 0.35})`;
                ctx.lineWidth = thickness * (0.45 + energy * 0.3);
                ctx.stroke();
            }
        }

        // Sun core — white-hot center
        const sunCore = ctx.createRadialGradient(
            -sunSize * 0.15, -sunSize * 0.15, sunSize * 0.05,
            0, 0, sunSize
        );
        sunCore.addColorStop(0, `hsla(${brassHue}, 30%, 100%, 1)`);
        sunCore.addColorStop(0.3, `hsla(${brassHue}, 70%, 90%, 0.95)`);
        sunCore.addColorStop(0.7, `hsla(${brassHue}, 90%, 65%, 0.85)`);
        sunCore.addColorStop(1, `hsla(${brassHue + 10}, 100%, 50%, 0.6)`);
        ctx.beginPath();
        ctx.arc(0, 0, sunSize, 0, Math.PI * 2);
        ctx.fillStyle = sunCore;
        ctx.fill();

        // Secondary star as subtle glow offset
        const sBSize = sunSize * 0.5;
        const sBGlow = ctx.createRadialGradient(sBx, sBy, sBSize * 0.1, sBx, sBy, sBSize * 2);
        sBGlow.addColorStop(0, `hsla(210, 60%, 85%, 0.5)`);
        sBGlow.addColorStop(0.5, `hsla(210, 50%, 70%, 0.15)`);
        sBGlow.addColorStop(1, `hsla(210, 40%, 60%, 0)`);
        ctx.beginPath();
        ctx.arc(sBx, sBy, sBSize * 2, 0, Math.PI * 2);
        ctx.fillStyle = sBGlow;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(sBx, sBy, sBSize, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(210, 50%, 95%, 0.7)`;
        ctx.fill();

        // =====================================================================
        // 7. RENDER PLANETS (back-to-front with stalks)
        // =====================================================================
        for (const item of renderList) {
            const { i, spec, discX, discY, planetX, planetY, bodySize } = item;
            const selfSpin = rot * (3.0 - i * 0.3);
            const [pH, pS, pL] = spec.color;

            // --- Stalk (vertical brass rod from disc to planet) ---
            ctx.beginPath();
            ctx.moveTo(discX, discY);
            ctx.lineTo(planetX, planetY);
            ctx.strokeStyle = `hsla(${brassHue}, ${brassSat}%, 50%, 0.6)`;
            ctx.lineWidth = thickness * 0.45;
            ctx.stroke();

            // Stalk top joint
            ctx.beginPath();
            ctx.arc(planetX, planetY + bodySize + 1, 2, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${brassHue}, ${brassSat}%, 55%, 0.5)`;
            ctx.fill();

            // --- Planet shadow on disc ---
            ctx.save();
            ctx.scale(1, tilt * 0.5);
            ctx.beginPath();
            ctx.arc(discX, discY / (tilt * 0.5), bodySize * 0.7, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(0, 0%, 0%, 0.12)`;
            ctx.fill();
            ctx.restore();

            // --- Planet sphere ---
            const pGrad = ctx.createRadialGradient(
                planetX - bodySize * 0.3, planetY - bodySize * 0.3, bodySize * 0.05,
                planetX, planetY, bodySize
            );
            pGrad.addColorStop(0, `hsla(${pH}, ${pS}%, ${pL + 20}%, 0.95)`);
            pGrad.addColorStop(0.5, `hsla(${pH}, ${pS}%, ${pL}%, 0.90)`);
            pGrad.addColorStop(1, `hsla(${pH}, ${pS - 10}%, ${pL - 15}%, 0.85)`);
            ctx.beginPath();
            ctx.arc(planetX, planetY, bodySize, 0, Math.PI * 2);
            ctx.fillStyle = pGrad;
            ctx.fill();

            // --- Planet-specific details ---

            // COLOSSUS (i=3): atmospheric bands + storm eye
            if (i === 3) {
                ctx.save();
                ctx.beginPath();
                ctx.arc(planetX, planetY, bodySize * 0.96, 0, Math.PI * 2);
                ctx.clip();
                for (let b = 0; b < 6; b++) {
                    const bandY = planetY - bodySize + (b + 0.5) * (bodySize * 2 / 6);
                    ctx.beginPath();
                    ctx.rect(planetX - bodySize, bandY, bodySize * 2, bodySize * 2 / 8);
                    ctx.fillStyle = `hsla(${25 + b * 5}, 50%, ${50 + b * 4}%, 0.15)`;
                    ctx.fill();
                }
                // Storm eye
                const stormPulse = 1.0 + energy * 0.3;
                ctx.beginPath();
                ctx.ellipse(
                    planetX + bodySize * 0.2, planetY + bodySize * 0.15,
                    bodySize * 0.2 * stormPulse, bodySize * 0.12 * stormPulse,
                    0.1, 0, Math.PI * 2
                );
                ctx.fillStyle = `hsla(10, 80%, 45%, 0.5)`;
                ctx.fill();
                ctx.restore();
            }

            // TERRA (i=1): cloud wisps
            if (i === 1) {
                ctx.save();
                ctx.beginPath();
                ctx.arc(planetX, planetY, bodySize * 0.93, 0, Math.PI * 2);
                ctx.clip();
                for (let cw = 0; cw < 3; cw++) {
                    const cwAngle = selfSpin * 0.05 + cw * 1.8 + this.seededRandom(seed + cw * 61) * 2;
                    ctx.beginPath();
                    ctx.arc(
                        planetX + Math.cos(cwAngle) * bodySize * 0.25,
                        planetY + Math.sin(cwAngle) * bodySize * 0.25,
                        bodySize * 0.35, cwAngle, cwAngle + 1.0
                    );
                    ctx.strokeStyle = `hsla(0, 0%, 100%, 0.2)`;
                    ctx.lineWidth = bodySize * 0.12;
                    ctx.stroke();
                }
                ctx.restore();
            }

            // ARIDUS (i=2): polar cap
            if (i === 2) {
                ctx.save();
                ctx.beginPath();
                ctx.arc(planetX, planetY, bodySize * 0.93, 0, Math.PI * 2);
                ctx.clip();
                ctx.beginPath();
                ctx.arc(planetX, planetY - bodySize * 0.65, bodySize * 0.45, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(0, 0%, 95%, 0.3)`;
                ctx.fill();
                ctx.restore();
            }

            // VULCAN (i=0): lava cracks
            if (i === 0) {
                ctx.save();
                ctx.beginPath();
                ctx.arc(planetX, planetY, bodySize * 0.93, 0, Math.PI * 2);
                ctx.clip();
                for (let c = 0; c < 4; c++) {
                    const cAngle = this.seededRandom(seed + c * 83) * Math.PI * 2 + selfSpin * 0.1;
                    const cLen = bodySize * (0.4 + this.seededRandom(seed + c * 97) * 0.4);
                    ctx.beginPath();
                    ctx.moveTo(
                        planetX + Math.cos(cAngle) * bodySize * 0.15,
                        planetY + Math.sin(cAngle) * bodySize * 0.15
                    );
                    ctx.lineTo(
                        planetX + Math.cos(cAngle + 0.3) * cLen,
                        planetY + Math.sin(cAngle + 0.3) * cLen
                    );
                    ctx.strokeStyle = `hsla(15, 100%, 55%, ${0.15 + energy * 0.25})`;
                    ctx.lineWidth = 0.8 + energy * 0.5;
                    ctx.stroke();
                }
                ctx.restore();
            }

            // TEMPEST (i=6): dark spot
            if (i === 6) {
                ctx.save();
                ctx.beginPath();
                ctx.arc(planetX, planetY, bodySize * 0.93, 0, Math.PI * 2);
                ctx.clip();
                ctx.beginPath();
                ctx.ellipse(
                    planetX + bodySize * 0.15, planetY - bodySize * 0.1,
                    bodySize * 0.15, bodySize * 0.08, 0.2, 0, Math.PI * 2
                );
                ctx.fillStyle = `hsla(240, 40%, 15%, 0.35)`;
                ctx.fill();
                ctx.restore();
            }

            // --- Rings (AURELIAN i=4, GLACIUS i=5) ---
            if (spec.ring) {
                const ringScale = i === 4 ? 0.25 : 0.12;
                const ringWidth = i === 4 ? bodySize * 0.15 : bodySize * 0.06;
                const ringOuterR = bodySize * (i === 4 ? 1.8 : 1.4);

                // Back half
                ctx.save();
                ctx.translate(planetX, planetY);
                ctx.scale(1, ringScale);
                ctx.beginPath();
                ctx.arc(0, 0, ringOuterR, 0, Math.PI);
                ctx.strokeStyle = i === 4
                    ? `hsla(42, 55%, 65%, 0.35)`
                    : `hsla(180, 40%, 60%, 0.20)`;
                ctx.lineWidth = ringWidth;
                ctx.stroke();
                if (i === 4) {
                    // Cassini gap
                    ctx.beginPath();
                    ctx.arc(0, 0, ringOuterR * 0.85, 0, Math.PI);
                    ctx.strokeStyle = `hsla(0, 0%, 0%, 0.12)`;
                    ctx.lineWidth = ringWidth * 0.2;
                    ctx.stroke();
                }
                ctx.restore();

                // Front half (draw after planet for proper overlap)
                ctx.save();
                ctx.translate(planetX, planetY);
                ctx.scale(1, ringScale);
                ctx.beginPath();
                ctx.arc(0, 0, ringOuterR, Math.PI, Math.PI * 2);
                ctx.strokeStyle = i === 4
                    ? `hsla(42, 55%, 65%, 0.35)`
                    : `hsla(180, 40%, 60%, 0.20)`;
                ctx.lineWidth = ringWidth;
                ctx.stroke();
                if (i === 4) {
                    ctx.beginPath();
                    ctx.arc(0, 0, ringOuterR * 0.85, Math.PI, Math.PI * 2);
                    ctx.strokeStyle = `hsla(0, 0%, 0%, 0.12)`;
                    ctx.lineWidth = ringWidth * 0.2;
                    ctx.stroke();
                    // Ring sparkle on beats
                    if (energy > 0.3) {
                        for (let sp = 0; sp < Math.floor(3 + density * 8); sp++) {
                            const spAngle = this.seededRandom(seed + sp * 43 + 4000) * Math.PI * 2;
                            const spR = ringOuterR * (0.75 + this.seededRandom(seed + sp * 67 + 4000) * 0.3);
                            ctx.beginPath();
                            ctx.arc(Math.cos(spAngle + rot * 2) * spR, Math.sin(spAngle + rot * 2) * spR, 0.8 + energy, 0, Math.PI * 2);
                            ctx.fillStyle = `hsla(50, 80%, 90%, ${energy * 0.5})`;
                            ctx.fill();
                        }
                    }
                }
                ctx.restore();
            }

            // --- Moons (small dots orbiting planet) ---
            const moonCounts = [0, 1, 0, 3, 2, 1, 1, 0];
            const moonCount = Math.min(moonCounts[i], Math.ceil(density * moonCounts[i]));
            for (let mi = 0; mi < moonCount; mi++) {
                const mOrbit = bodySize * (1.8 + mi * 0.7);
                const mAngle = rot * (4 - mi * 0.8) + this.seededRandom(seed + i * 1000 + mi) * Math.PI * 2;
                const mX = planetX + Math.cos(mAngle) * mOrbit;
                const mY = planetY + Math.sin(mAngle) * mOrbit * 0.5; // flattened orbit
                const mSize = bodySize * 0.15;
                const mGrad = ctx.createRadialGradient(
                    mX - mSize * 0.3, mY - mSize * 0.3, mSize * 0.1,
                    mX, mY, mSize
                );
                mGrad.addColorStop(0, `hsla(0, 5%, 75%, 0.85)`);
                mGrad.addColorStop(1, `hsla(0, 5%, 50%, 0.6)`);
                ctx.beginPath();
                ctx.arc(mX, mY, mSize, 0, Math.PI * 2);
                ctx.fillStyle = mGrad;
                ctx.fill();
            }
        }

        // =====================================================================
        // 8. ASTEROID BELT (density > 0.8)
        // =====================================================================
        if (density > 0.8) {
            const beltSimR = 2.5; // between Aridus and Colossus
            const asteroidCount = Math.floor((density - 0.8) * 50);
            for (let a = 0; a < asteroidCount; a++) {
                const aPhase = this.seededRandom(seed + a * 173 + 8000) * Math.PI * 2;
                const aSpeed = 0.3 + this.seededRandom(seed + a * 211 + 8001) * 0.15;
                const aR = beltSimR * simScale * (0.9 + this.seededRandom(seed + a * 97 + 8002) * 0.2);
                const aAngle = aPhase + rot * aSpeed;
                const ax = Math.cos(aAngle) * aR;
                const ay = Math.sin(aAngle) * aR * tilt; // on disc plane
                const aSize = 0.4 + this.seededRandom(seed + a * 53 + 8003) * 1.0;
                ctx.beginPath();
                ctx.arc(ax, ay, aSize, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${brassHue}, 25%, 50%, 0.25)`;
                ctx.fill();
            }
        }

        // =====================================================================
        // 9. BASS SHOCKWAVE RINGS
        // =====================================================================
        if (energy > 0.5) {
            const shockCycle = (rot * 8) % (Math.PI * 2);
            const frac = shockCycle / (Math.PI * 2);
            const shockR = frac * discRadius * 0.7;
            const shockAlpha = (1 - frac) * energy * 0.25;
            ctx.save();
            ctx.scale(1, tilt);
            ctx.beginPath();
            ctx.arc(0, 0, shockR, 0, Math.PI * 2);
            ctx.strokeStyle = `hsla(${brassHue}, 80%, 75%, ${shockAlpha})`;
            ctx.lineWidth = thickness * (0.6 * (1 - frac) + 0.15);
            ctx.stroke();
            ctx.restore();
        }

        // =====================================================================
        // 10. WANDERER'S COMET TAIL (if visible)
        // =====================================================================
        if (visiblePlanets >= 8) {
            const wp = planets[7];
            const wpx = wp.x * simScale;
            const wpy = wp.y * simScale * tilt;
            const wStalk = canvasBase * 0.010 * orbitFactor * 1.8 + 8;
            const wpPlanetY = wpy - wStalk;
            const awayAngle = Math.atan2(wpy, wpx);
            const tailLen = canvasBase * 0.010 * orbitFactor * 4;
            ctx.beginPath();
            ctx.moveTo(wpx, wpPlanetY);
            ctx.quadraticCurveTo(
                wpx + Math.cos(awayAngle) * tailLen * 0.5 + Math.sin(rot * 5) * 3,
                wpPlanetY + Math.sin(awayAngle) * tailLen * 0.3,
                wpx + Math.cos(awayAngle) * tailLen,
                wpPlanetY + Math.sin(awayAngle) * tailLen * 0.2
            );
            ctx.strokeStyle = `hsla(200, 40%, 80%, 0.10)`;
            ctx.lineWidth = thickness * 0.35;
            ctx.stroke();
        }

        ctx.restore();
    }

    /**
     * Quark background — Quantum foam with Planck-scale jitter and vacuum fluctuations
     * 3 parallax layers: quantum foam (far), interference fringes (mid), virtual particles (near)
     */
    renderQuarkBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const maxDim = Math.max(width, height) * 0.75;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const sat = config.saturation;

        // Cherenkov blue base
        const cherenkovHue = 210;

        // --- FAR LAYER (0.2x): Quantum foam — tiny flickering bubbles ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 0.2);

        const foamCount = 30;
        for (let i = 0; i < foamCount; i++) {
            const seed = i * 53 + 31;
            const dist = maxDim * (0.05 + this.seededRandom(seed) * 0.95);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            // Pop in/out: sinusoidal life cycle
            const lifePhase = Math.sin(rot * (3 + this.seededRandom(seed + 2) * 4) + this.seededRandom(seed + 3) * Math.PI * 2);
            if (lifePhase < 0) continue; // "popped out of existence"
            const size = (1 + this.seededRandom(seed + 4) * 2) * lifePhase;
            const alpha = (0.03 + reactivity * 0.05) * lifePhase;

            const fx = Math.cos(angle) * dist;
            const fy = Math.sin(angle) * dist;

            ctx.beginPath();
            ctx.arc(fx, fy, size, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${cherenkovHue}, ${sat * 0.4}%, ${50 + this.seededRandom(seed + 5) * 20}%, ${alpha})`;
            ctx.fill();
        }
        ctx.restore();

        // --- MID LAYER (−0.5x): Double-slit interference fringes ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(-rot * 0.5);

        const fringeCount = 12;
        const fringeSpacing = maxDim * 0.08;
        for (let i = 0; i < fringeCount; i++) {
            const ringR = fringeSpacing * (i + 1);
            // Interference: bright/dark alternation based on phase
            const phase = Math.sin(rot * 2 + i * 1.5);
            const fringeAlpha = (0.02 + reactivity * 0.04) * Math.max(0, phase);

            if (fringeAlpha < 0.005) continue;

            ctx.beginPath();
            ctx.arc(0, 0, ringR, 0, Math.PI * 2);
            ctx.strokeStyle = `hsla(${cherenkovHue}, ${sat * 0.5}%, 60%, ${fringeAlpha})`;
            ctx.lineWidth = 2 + energy * reactivity * 2;
            ctx.stroke();
        }
        ctx.restore();

        // --- NEAR LAYER (1.2x): Virtual particle pairs — pop in and annihilate ---
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rot * 1.2);

        const pairCount = 8;
        for (let i = 0; i < pairCount; i++) {
            const seed = i * 67 + 211;
            const dist = maxDim * (0.15 + this.seededRandom(seed) * 0.65);
            const angle = this.seededRandom(seed + 1) * Math.PI * 2;
            const phase = this.seededRandom(seed + 2) * Math.PI * 2;
            const life = Math.sin(rot * (2 + this.seededRandom(seed + 3) * 3) + phase);
            if (life < 0.1) continue;

            const separation = 4 + life * 6;
            const perpAngle = angle + Math.PI / 2;
            const cx1 = Math.cos(angle) * dist + Math.cos(perpAngle) * separation;
            const cy1 = Math.sin(angle) * dist + Math.sin(perpAngle) * separation;
            const cx2 = Math.cos(angle) * dist - Math.cos(perpAngle) * separation;
            const cy2 = Math.sin(angle) * dist - Math.sin(perpAngle) * separation;
            const pSize = 2 + life * 2;
            const alpha = (0.04 + reactivity * 0.08) * life;

            // Particle
            ctx.beginPath();
            ctx.arc(cx1, cy1, pSize, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${cherenkovHue}, ${sat * 0.7}%, 65%, ${alpha})`;
            ctx.fill();

            // Anti-particle (complementary hue)
            ctx.beginPath();
            ctx.arc(cx2, cy2, pSize, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${(cherenkovHue + 180) % 360}, ${sat * 0.6}%, 55%, ${alpha})`;
            ctx.fill();

            // Annihilation line
            ctx.beginPath();
            ctx.moveTo(cx1, cy1);
            ctx.lineTo(cx2, cy2);
            ctx.strokeStyle = `hsla(${cherenkovHue}, ${sat * 0.3}%, 50%, ${alpha * 0.5})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }
        ctx.restore();
    }

    /**
     * Quark style — Quantum field theory visualizer. Shows quarks and electrons as disturbances
     * in living, breathing quantum fields. The strong field (gluon flux tubes), electromagnetic
     * field (virtual photon exchange), weak force (boson flares), and the Higgs condensate
     * are all visible as interconnected, pulsing substrates of reality.
     */
    renderQuarkStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const mirrors = config.mirrors;
        const rot = this.accumulatedRotation;
        const orbitFactor = config.orbitRadius / 200;
        const sat = config.saturation;

        const cherenkovHue = 210;
        const colorChargePhase = rot * 0.5;
        // Quark count driven by numSides (3-8), with evenly-spaced color charges
        const quarkCount = Math.max(3, Math.min(8, numSides));
        const qHues = [];
        for (let qi = 0; qi < quarkCount; qi++) {
            qHues.push((qi * 360 / quarkCount) % 360);
        }
        const fieldExtent = radius * 0.7 * orbitFactor;

        ctx.save();
        ctx.translate(centerX, centerY);

        // =====================================================
        // LAYER 0: HIGGS CONDENSATE — the fabric everything swims in
        // A living mesh that undulates, giving mass to all particles.
        // Slow, pervasive, always present. The "floor" of reality.
        // =====================================================
        ctx.save();
        ctx.rotate(rot * 0.03); // barely drifts
        const higgsNodes = 10 + Math.floor(harmonic * 4);
        const higgsCellSize = fieldExtent * 2 / higgsNodes;
        const higgsAlpha = 0.025 + harmonic * 0.025;

        for (let i = 0; i < higgsNodes; i++) {
            for (let j = 0; j < higgsNodes; j++) {
                const hSeed = seed + i * 97 + j * 53;
                if (this.seededRandom(hSeed) > 0.4) continue; // sparse

                const bx = (i - higgsNodes / 2) * higgsCellSize;
                const by = (j - higgsNodes / 2) * higgsCellSize;
                const dist = Math.sqrt(bx * bx + by * by);
                if (dist > fieldExtent) continue;

                // Field displacement — the condensate breathes
                const disp = Math.sin(dist * 0.02 + rot * 0.5) * higgsCellSize * 0.15
                    + Math.sin(rot * 0.3 + i * 0.4 + j * 0.3) * higgsCellSize * 0.08 * harmonic;
                const nx = bx + disp;
                const ny = by + Math.cos(dist * 0.02 + rot * 0.4) * higgsCellSize * 0.12;

                // Nearby nodes — draw a faint connection
                if (i < higgsNodes - 1) {
                    const ni = i + 1;
                    const nbx = (ni - higgsNodes / 2) * higgsCellSize;
                    const nby = by;
                    const nDisp = Math.sin(Math.sqrt(nbx * nbx + nby * nby) * 0.02 + rot * 0.5) * higgsCellSize * 0.15;
                    ctx.beginPath();
                    ctx.moveTo(nx, ny);
                    ctx.lineTo(nbx + nDisp, nby + Math.cos(Math.sqrt(nbx * nbx + nby * nby) * 0.02 + rot * 0.4) * higgsCellSize * 0.12);
                    ctx.strokeStyle = `hsla(45, ${sat * 0.2}%, 40%, ${higgsAlpha * (1 - dist / fieldExtent)})`;
                    ctx.lineWidth = 0.4;
                    ctx.stroke();
                }
                if (j < higgsNodes - 1) {
                    const nj = j + 1;
                    const nbx = bx;
                    const nby = (nj - higgsNodes / 2) * higgsCellSize;
                    const nDisp = Math.sin(Math.sqrt(nbx * nbx + nby * nby) * 0.02 + rot * 0.5) * higgsCellSize * 0.15;
                    ctx.beginPath();
                    ctx.moveTo(nx, ny);
                    ctx.lineTo(nbx + nDisp, nby + Math.cos(Math.sqrt(nbx * nbx + nby * nby) * 0.02 + rot * 0.4) * higgsCellSize * 0.12);
                    ctx.strokeStyle = `hsla(45, ${sat * 0.2}%, 40%, ${higgsAlpha * (1 - dist / fieldExtent)})`;
                    ctx.lineWidth = 0.4;
                    ctx.stroke();
                }
            }
        }
        ctx.restore();

        // =====================================================
        // LAYER 1: STRONG FORCE FIELD — Color flux tubes radiating from quarks
        // Confined color field lines that don't spread out like EM —
        // they form "tubes" that get brighter as they stretch. Confinement.
        // =====================================================
        const quarkOrbit = radius * 0.09 * orbitFactor * (0.7 + energy * 0.5);
        const quarkPositions = [];
        for (let q = 0; q < quarkCount; q++) {
            const phase = (Math.PI * 2 * q) / quarkCount;
            const vibrate = Math.sin(rot * 3 + phase * 2) * quarkOrbit * 0.25 * harmonic;
            const jitter = Math.sin(rot * 11 + q * 5) * quarkOrbit * 0.05;
            const qx = Math.cos(phase + rot * 0.4) * (quarkOrbit + vibrate + jitter);
            const qy = Math.sin(phase + rot * 0.4) * (quarkOrbit + vibrate + jitter);
            quarkPositions.push({ x: qx, y: qy });
        }

        // Strong field flux tubes — thick, glowing confined tubes between quarks
        for (let q = 0; q < quarkCount; q++) {
            const p0 = quarkPositions[q];
            const p1 = quarkPositions[(q + 1) % quarkCount];
            const chargeIdx = (q + Math.floor(colorChargePhase)) % quarkCount;
            const tubeHue = qHues[chargeIdx];
            const dx = p1.x - p0.x;
            const dy = p1.y - p0.y;
            const len = Math.sqrt(dx * dx + dy * dy) || 1;
            const perpX = -dy / len;
            const perpY = dx / len;

            // Flux tube: filled region with pulsing width
            const tubeWidth = quarkOrbit * 0.15 * (1 + energy * 0.6);
            const segs = 20;

            // Draw flux tube as filled shape
            ctx.beginPath();
            for (let s = 0; s <= segs; s++) {
                const t = s / segs;
                const mx = p0.x + dx * t;
                const my = p0.y + dy * t;
                const wave = Math.sin(t * Math.PI * 6 + rot * 4) * tubeWidth * 0.3;
                ctx.lineTo(mx + perpX * (tubeWidth + wave), my + perpY * (tubeWidth + wave));
            }
            for (let s = segs; s >= 0; s--) {
                const t = s / segs;
                const mx = p0.x + dx * t;
                const my = p0.y + dy * t;
                const wave = Math.sin(t * Math.PI * 6 + rot * 4) * tubeWidth * 0.3;
                ctx.lineTo(mx - perpX * (tubeWidth + wave), my - perpY * (tubeWidth + wave));
            }
            ctx.closePath();
            ctx.fillStyle = `hsla(${tubeHue}, ${sat * 0.6}%, 25%, ${0.08 + energy * 0.1})`;
            ctx.fill();

            // Bright core string — double helix inside the tube
            for (let strand = 0; strand < 2; strand++) {
                ctx.beginPath();
                ctx.moveTo(p0.x, p0.y);
                const dir = strand === 0 ? 1 : -1;
                for (let g = 1; g <= segs; g++) {
                    const t = g / segs;
                    const mx = p0.x + dx * t;
                    const my = p0.y + dy * t;
                    const wave = Math.sin(t * Math.PI * 5 + rot * 3.5 + strand * Math.PI) * tubeWidth * 0.5 * dir;
                    ctx.lineTo(mx + perpX * wave, my + perpY * wave);
                }
                const nextChargeIdx = ((q + 1) + Math.floor(colorChargePhase)) % quarkCount;
                const strandHue = strand === 0 ? tubeHue : qHues[nextChargeIdx];
                ctx.strokeStyle = `hsla(${strandHue}, ${sat * 0.8}%, 60%, ${0.2 + energy * 0.3})`;
                ctx.lineWidth = thickness * 0.3;
                ctx.stroke();
            }

            // Running "gluon pulse" along the tube
            const pulseT = ((rot * 2 + q * 2) % (Math.PI * 2)) / (Math.PI * 2);
            const pulseX = p0.x + dx * pulseT;
            const pulseY = p0.y + dy * pulseT;
            const pulseR = tubeWidth * (1.5 + energy * 0.5);
            const pulseGrad = ctx.createRadialGradient(pulseX, pulseY, 0, pulseX, pulseY, pulseR);
            pulseGrad.addColorStop(0, `hsla(${tubeHue}, ${sat * 0.9}%, 75%, ${0.3 + energy * 0.3})`);
            pulseGrad.addColorStop(0.5, `hsla(${tubeHue}, ${sat * 0.6}%, 45%, ${0.08 + energy * 0.08})`);
            pulseGrad.addColorStop(1, `hsla(${tubeHue}, ${sat * 0.3}%, 25%, 0)`);
            ctx.beginPath();
            ctx.arc(pulseX, pulseY, pulseR, 0, Math.PI * 2);
            ctx.fillStyle = pulseGrad;
            ctx.fill();
        }

        // === QUARK CORES — with superposition echoes ===
        for (let q = 0; q < quarkCount; q++) {
            const { x: qx, y: qy } = quarkPositions[q];
            const qSize = radius * 0.025 * orbitFactor * (0.8 + energy * 0.3);
            const chargeIdx = (q + Math.floor(colorChargePhase)) % quarkCount;
            const qHue = qHues[chargeIdx];

            // Superposition echoes
            for (let echo = 3; echo >= 1; echo--) {
                const phase = (Math.PI * 2 * q) / quarkCount;
                const echoRot = rot * 0.4 - echo * 0.18;
                const echoVib = Math.sin((rot - echo * 0.35) * 3 + phase * 2) * quarkOrbit * 0.25 * harmonic;
                const ex = Math.cos(phase + echoRot) * (quarkOrbit + echoVib);
                const ey = Math.sin(phase + echoRot) * (quarkOrbit + echoVib);
                ctx.beginPath();
                ctx.arc(ex, ey, qSize * (1.8 + echo * 0.6), 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${qHue}, ${sat * 0.4}%, 45%, ${0.04 / echo})`;
                ctx.fill();
            }

            // Deep glow
            const qGlow = ctx.createRadialGradient(qx, qy, 0, qx, qy, qSize * 5);
            qGlow.addColorStop(0, `hsla(${qHue}, ${sat * 0.9}%, 80%, ${0.7 + energy * 0.3})`);
            qGlow.addColorStop(0.2, `hsla(${qHue}, ${sat * 0.7}%, 55%, ${0.3 + energy * 0.15})`);
            qGlow.addColorStop(0.6, `hsla(${qHue}, ${sat * 0.4}%, 30%, 0.04)`);
            qGlow.addColorStop(1, `hsla(${qHue}, ${sat * 0.2}%, 15%, 0)`);
            ctx.beginPath();
            ctx.arc(qx, qy, qSize * 5, 0, Math.PI * 2);
            ctx.fillStyle = qGlow;
            ctx.fill();

            // Core
            ctx.beginPath();
            ctx.arc(qx, qy, qSize, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${qHue}, ${sat * 0.6}%, 92%, 0.95)`;
            ctx.fill();
        }

        // White-light core when color charges balance
        const coreBalance = Math.abs(Math.sin(colorChargePhase * Math.PI / (quarkCount / 2)));
        if (coreBalance > 0.8) {
            const whiteGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, quarkOrbit * 0.7);
            whiteGrad.addColorStop(0, `hsla(0, 0%, 100%, ${(coreBalance - 0.8) * 3})`);
            whiteGrad.addColorStop(0.4, `hsla(${cherenkovHue}, 20%, 85%, ${(coreBalance - 0.8) * 0.5})`);
            whiteGrad.addColorStop(1, `hsla(0, 0%, 100%, 0)`);
            ctx.beginPath();
            ctx.arc(0, 0, quarkOrbit * 0.7, 0, Math.PI * 2);
            ctx.fillStyle = whiteGrad;
            ctx.fill();
        }

        // =====================================================
        // LAYER 2: PER-SEGMENT — EM field, electrons, weak force, vacuum
        // =====================================================
        for (let m = 0; m < mirrors; m++) {
            const mirrorAngle = (Math.PI * 2 * m) / mirrors;

            ctx.save();
            ctx.rotate(mirrorAngle);
            const segAngle = Math.PI / mirrors;

            // Eerie drift
            ctx.rotate(Math.sin(rot * 0.25 + m * 1.7) * 0.015);

            // --- ELECTROMAGNETIC FIELD LINES ---
            // Radial field lines emanating from center outward through segment,
            // distorted by the particles they pass near. These are the "force"
            // that connects quarks to electrons — visible communication.
            const fieldLineCount = 3 + Math.floor(this.seededRandom(seed + m * 7 + 400) * 2);
            for (let fl = 0; fl < fieldLineCount; fl++) {
                const flSeed = seed + 400 + m * 20 + fl * 13;
                const flAngle = segAngle * (0.1 + fl * 0.8 / fieldLineCount);
                const flSteps = 25;
                const flHue = (cherenkovHue + this.seededRandom(flSeed) * 30) % 360;

                ctx.beginPath();
                for (let s = 0; s <= flSteps; s++) {
                    const t = s / flSteps;
                    const baseR = radius * (0.05 + t * 0.55) * orbitFactor;
                    // Field lines curve and warp near particles
                    const distortion = Math.sin(t * Math.PI * 3 + rot * 1.5 + fl * 1.1) * radius * 0.012 * harmonic
                        + Math.sin(t * Math.PI * 7 + rot * 3) * radius * 0.005 * energy;
                    const angle = flAngle + distortion / baseR;
                    const x = Math.cos(angle) * baseR;
                    const y = Math.sin(angle) * baseR;
                    s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                // Fade out toward edges
                ctx.strokeStyle = `hsla(${flHue}, ${sat * 0.35}%, 45%, ${0.04 + harmonic * 0.05 + energy * 0.03})`;
                ctx.lineWidth = 0.5 + energy * 0.5;
                ctx.stroke();
            }

            // --- VIRTUAL PHOTON EXCHANGE: streaking bolts from quarks to electrons ---
            // These are the "communication" — force carriers zipping between particles
            const photonCount = 1 + Math.floor(energy * 2 + brightness * 1);
            for (let ph = 0; ph < photonCount; ph++) {
                const phSeed = seed + 500 + m * 30 + ph * 17;
                // Photon travels from center outward at varying speed
                const speed = 1.5 + this.seededRandom(phSeed) * 2;
                const phLife = ((rot * speed + this.seededRandom(phSeed + 1) * Math.PI * 2) % (Math.PI * 2)) / (Math.PI * 2);
                const phAngle = segAngle * (0.15 + this.seededRandom(phSeed + 2) * 0.7);
                const startR = quarkOrbit * 1.5;
                const endR = radius * (0.35 + this.seededRandom(phSeed + 3) * 0.2) * orbitFactor;
                const phR = startR + (endR - startR) * phLife;
                const phX = Math.cos(phAngle) * phR;
                const phY = Math.sin(phAngle) * phR;

                // Photon is a tiny wavy streak
                const phLen = radius * 0.03 * orbitFactor;
                const phHue = cherenkovHue;
                const phAlpha = Math.sin(phLife * Math.PI) * (0.3 + energy * 0.4); // fade in/out

                if (phAlpha > 0.02) {
                    // Wavy body
                    ctx.beginPath();
                    const waveSteps = 6;
                    for (let w = 0; w <= waveSteps; w++) {
                        const wt = w / waveSteps;
                        const wr = phR - phLen * wt;
                        const wAngle = phAngle + Math.sin(wt * Math.PI * 3 + rot * 8) * 0.02;
                        const wx = Math.cos(wAngle) * wr;
                        const wy = Math.sin(wAngle) * wr;
                        w === 0 ? ctx.moveTo(wx, wy) : ctx.lineTo(wx, wy);
                    }
                    ctx.strokeStyle = `hsla(${phHue}, ${sat * 0.8}%, 75%, ${phAlpha})`;
                    ctx.lineWidth = 1 + energy;
                    ctx.stroke();

                    // Head glow
                    const headGrad = ctx.createRadialGradient(phX, phY, 0, phX, phY, 3 + energy * 2);
                    headGrad.addColorStop(0, `hsla(${phHue}, ${sat * 0.9}%, 90%, ${phAlpha * 0.8})`);
                    headGrad.addColorStop(1, `hsla(${phHue}, ${sat * 0.5}%, 50%, 0)`);
                    ctx.beginPath();
                    ctx.arc(phX, phY, 3 + energy * 2, 0, Math.PI * 2);
                    ctx.fillStyle = headGrad;
                    ctx.fill();
                }
            }

            // --- ELECTRON SHELLS with Cherenkov wake ---
            const shellCount = 2 + Math.floor(this.seededRandom(seed + m * 3 + 500) * 2);
            for (let sh = 0; sh < shellCount; sh++) {
                const shSeed = seed + 200 + m * 30 + sh * 19;
                const baseShellR = radius * (0.22 + sh * 0.14) * orbitFactor;
                const leapTrigger = Math.sin(rot * 1.5 + this.seededRandom(shSeed) * Math.PI * 2);
                const shellR = leapTrigger > 0.7
                    ? baseShellR * 1.35
                    : leapTrigger < -0.7
                        ? baseShellR * 0.75
                        : baseShellR;
                const shellHue = (cherenkovHue + sh * 30) % 360;

                // Shell arc — probability amplitude wave
                ctx.beginPath();
                const arcSteps = 35;
                for (let a = 0; a <= arcSteps; a++) {
                    const angle = (segAngle * a) / arcSteps;
                    const psi = Math.sin(angle * (6 + sh * 2) + rot * 2 + sh * 2);
                    const wobble = psi * shellR * 0.025 * (1.2 - energy * 0.5);
                    const r = shellR + wobble;
                    const x = Math.cos(angle) * r;
                    const y = Math.sin(angle) * r;
                    a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                ctx.strokeStyle = `hsla(${shellHue}, ${sat * 0.5}%, ${40 + brightness * 20}%, ${0.08 + harmonic * 0.15})`;
                ctx.lineWidth = thickness * 0.25;
                ctx.stroke();

                // Electron with deep Cherenkov cone
                const eAngle = segAngle * (0.15 + 0.7 * ((Math.sin(rot * (1.8 + sh * 0.4)) + 1) / 2));
                const ex = Math.cos(eAngle) * shellR;
                const ey = Math.sin(eAngle) * shellR;

                // Cherenkov cone — v-shaped wake behind electron
                const coneLen = radius * 0.05 * orbitFactor * (1 + energy * 0.5);
                const coneSpread = 0.15;
                for (let c = 0; c < 2; c++) {
                    const cDir = c === 0 ? 1 : -1;
                    const trailAngle = eAngle - 0.12; // behind direction of motion
                    const tipR = shellR;
                    const baseAngle = trailAngle + coneSpread * cDir;
                    ctx.beginPath();
                    ctx.moveTo(ex, ey);
                    ctx.lineTo(
                        Math.cos(baseAngle) * (tipR - coneLen),
                        Math.sin(baseAngle) * (tipR - coneLen)
                    );
                    ctx.strokeStyle = `hsla(${cherenkovHue}, ${sat * 0.7}%, 60%, ${0.06 + energy * 0.08})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }

                // Cherenkov glow
                const eGlow = ctx.createRadialGradient(ex, ey, 0, ex, ey, 8 + energy * 5);
                eGlow.addColorStop(0, `hsla(${cherenkovHue}, ${sat * 0.95}%, 85%, ${0.7 + energy * 0.3})`);
                eGlow.addColorStop(0.3, `hsla(${cherenkovHue}, ${sat * 0.8}%, 60%, ${0.2 + energy * 0.1})`);
                eGlow.addColorStop(0.7, `hsla(${cherenkovHue}, ${sat * 0.5}%, 35%, 0.03)`);
                eGlow.addColorStop(1, `hsla(${cherenkovHue}, ${sat * 0.3}%, 20%, 0)`);
                ctx.beginPath();
                ctx.arc(ex, ey, 8 + energy * 5, 0, Math.PI * 2);
                ctx.fillStyle = eGlow;
                ctx.fill();

                // Electron core
                ctx.beginPath();
                ctx.arc(ex, ey, 1.8 + energy * 0.8, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${cherenkovHue}, ${sat * 0.6}%, 93%, 0.95)`;
                ctx.fill();

                // --- PHOTON ABSORPTION/EMISSION at electron ---
                // When a virtual photon reaches the electron, show a brief radial burst
                const burstPhase = Math.sin(rot * 3 + sh * 2 + m * 1.3);
                if (burstPhase > 0.7 && energy > 0.2) {
                    const burstAlpha = (burstPhase - 0.7) * 2 * energy;
                    const burstR = 4 + (burstPhase - 0.7) * radius * 0.04;
                    const burstCount = 5;
                    for (let b = 0; b < burstCount; b++) {
                        const bAngle = (Math.PI * 2 * b) / burstCount + rot * 2;
                        const bLen = burstR * (1 + this.seededRandom(seed + m * 5 + sh * 3 + b) * 0.5);
                        ctx.beginPath();
                        ctx.moveTo(ex, ey);
                        ctx.lineTo(
                            ex + Math.cos(bAngle) * bLen,
                            ey + Math.sin(bAngle) * bLen
                        );
                        ctx.strokeStyle = `hsla(${cherenkovHue}, ${sat * 0.8}%, 80%, ${burstAlpha * 0.4})`;
                        ctx.lineWidth = 0.6;
                        ctx.stroke();
                    }
                }
            }

            // --- WEAK FORCE BOSON FLARES ---
            // Rare, heavy, slow-moving W/Z bosons that appear as expanding rings
            // between quarks and leptons. They mediate transformation.
            const weakSeed = seed + 700 + m * 13;
            const weakPhase = Math.sin(rot * 0.4 + this.seededRandom(weakSeed) * Math.PI * 2);
            if (weakPhase > 0.6) {
                const wAlpha = (weakPhase - 0.6) * 0.7;
                const wAngle = segAngle * (0.3 + this.seededRandom(weakSeed + 1) * 0.4);
                const wDist = radius * (0.15 + this.seededRandom(weakSeed + 2) * 0.15) * orbitFactor;
                const wx = Math.cos(wAngle) * wDist;
                const wy = Math.sin(wAngle) * wDist;
                const wExpand = (weakPhase - 0.6) * radius * 0.08 * orbitFactor;

                // Expanding ring — heavy boson propagating
                ctx.beginPath();
                ctx.arc(wx, wy, wExpand, 0, Math.PI * 2);
                ctx.strokeStyle = `hsla(50, ${sat * 0.7}%, 65%, ${wAlpha * 0.6})`;
                ctx.lineWidth = 2 + energy * 2;
                ctx.stroke();

                // Inner transformation flash
                const wFlash = ctx.createRadialGradient(wx, wy, 0, wx, wy, wExpand * 0.6);
                wFlash.addColorStop(0, `hsla(50, ${sat * 0.8}%, 80%, ${wAlpha * 0.4})`);
                wFlash.addColorStop(1, `hsla(50, ${sat * 0.5}%, 40%, 0)`);
                ctx.beginPath();
                ctx.arc(wx, wy, wExpand * 0.6, 0, Math.PI * 2);
                ctx.fillStyle = wFlash;
                ctx.fill();
            }

            // --- VACUUM POLARIZATION near quarks ---
            // Close to the quark center, virtual pairs appear and vanish,
            // showing the vacuum is alive and screening the color charge
            const vpCount = 3 + Math.floor(energy * 3);
            for (let vp = 0; vp < vpCount; vp++) {
                const vpSeed = seed + 800 + m * 20 + vp * 11;
                const vpLife = Math.sin(rot * (3 + this.seededRandom(vpSeed) * 3) + this.seededRandom(vpSeed + 1) * Math.PI * 2);
                if (vpLife < 0.2) continue;

                const vpAngle = segAngle * (0.05 + this.seededRandom(vpSeed + 2) * 0.4);
                const vpDist = radius * (0.05 + this.seededRandom(vpSeed + 3) * 0.1) * orbitFactor;
                const vpX = Math.cos(vpAngle) * vpDist;
                const vpY = Math.sin(vpAngle) * vpDist;
                const sep = 2 + vpLife * 3;
                const vpAlpha = (vpLife - 0.2) * 0.25;

                // Particle-antiparticle pair
                ctx.beginPath();
                ctx.arc(vpX + sep, vpY, 1 + vpLife, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${cherenkovHue}, ${sat * 0.6}%, 65%, ${vpAlpha})`;
                ctx.fill();
                ctx.beginPath();
                ctx.arc(vpX - sep, vpY, 1 + vpLife, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${(cherenkovHue + 180) % 360}, ${sat * 0.5}%, 55%, ${vpAlpha})`;
                ctx.fill();
            }

            // --- PROBABILITY WAVE (ψ) — the wave function ripple through field ---
            const psiR = radius * 0.35 * orbitFactor;
            const psiSegs = 40;
            const psiAlpha = 0.04 + harmonic * 0.06;
            ctx.beginPath();
            for (let w = 0; w <= psiSegs; w++) {
                const angle = (segAngle * w) / psiSegs;
                const psi = Math.sin(angle * (8 + brightness * 4) + rot * 1.5);
                const r = psiR + psi * radius * 0.018 * (1 + harmonic * 0.6);
                const x = Math.cos(angle) * r;
                const y = Math.sin(angle) * r;
                w === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.strokeStyle = `hsla(${(cherenkovHue + 30) % 360}, ${sat * 0.4}%, 50%, ${psiAlpha})`;
            ctx.lineWidth = thickness * 0.2;
            ctx.stroke();

            ctx.restore();
        }

        // =====================================================
        // LAYER 3: FIELD RESONANCE WEB — when energy is high, standing waves
        // connect all particles into one coherent web. The universe talks to itself.
        // =====================================================
        if (energy > 0.25 || harmonic > 0.4) {
            const webAlpha = Math.max(energy - 0.25, 0) * 0.6 + Math.max(harmonic - 0.4, 0) * 0.3;
            const webNodes = [];
            // Collect quark positions
            for (const qp of quarkPositions) webNodes.push(qp);
            // Add electron approximate positions for web connections
            for (let m = 0; m < Math.min(mirrors, 6); m++) {
                const mAngle = (Math.PI * 2 * m) / mirrors;
                const segAngle = Math.PI / mirrors;
                const shellR = radius * 0.28 * orbitFactor;
                const eAngle = mAngle + segAngle * 0.5;
                webNodes.push({
                    x: Math.cos(eAngle) * shellR,
                    y: Math.sin(eAngle) * shellR
                });
            }

            // Draw web connections with standing wave patterns
            for (let i = 0; i < webNodes.length; i++) {
                for (let j = i + 1; j < webNodes.length; j++) {
                    const wSeed = seed + i * 100 + j * 7;
                    if (this.seededRandom(wSeed) > 0.35) continue;

                    const n0 = webNodes[i];
                    const n1 = webNodes[j];
                    const wdx = n1.x - n0.x;
                    const wdy = n1.y - n0.y;
                    const wLen = Math.sqrt(wdx * wdx + wdy * wdy) || 1;
                    const wPerpX = -wdy / wLen;
                    const wPerpY = wdx / wLen;

                    // Standing wave between nodes
                    ctx.beginPath();
                    const wSteps = 16;
                    for (let s = 0; s <= wSteps; s++) {
                        const t = s / wSteps;
                        const mx = n0.x + wdx * t;
                        const my = n0.y + wdy * t;
                        // Standing wave: sin(n*pi*t) — nodes at endpoints
                        const standingWave = Math.sin(t * Math.PI * 3) * Math.sin(rot * 3 + i + j) * wLen * 0.04;
                        ctx.lineTo(mx + wPerpX * standingWave, my + wPerpY * standingWave);
                    }
                    const webHue = (cherenkovHue + i * 20) % 360;
                    ctx.strokeStyle = `hsla(${webHue}, ${sat * 0.5}%, 55%, ${webAlpha * 0.15})`;
                    ctx.lineWidth = 0.5 + energy * 0.5;
                    ctx.stroke();
                }
            }
        }

        // === Central interference fringes ===
        const fringeRings = 8;
        for (let f = 0; f < fringeRings; f++) {
            const fR = radius * (0.07 + f * 0.028) * orbitFactor;
            const phase = Math.cos(rot * 2 + f * Math.PI);
            const fAlpha = Math.max(0, phase) * (0.08 + energy * 0.18);
            if (fAlpha < 0.01) continue;
            ctx.beginPath();
            ctx.arc(0, 0, fR, 0, Math.PI * 2);
            ctx.strokeStyle = `hsla(${cherenkovHue}, ${sat * 0.6}%, ${50 + brightness * 20}%, ${fAlpha})`;
            ctx.lineWidth = thickness * (0.2 + energy * 0.25);
            ctx.stroke();
        }

        // === Deep uncertainty haze ===
        const outerR = radius * 0.65 * orbitFactor;
        const haze1 = ctx.createRadialGradient(0, 0, outerR * 0.5, 0, 0, outerR);
        haze1.addColorStop(0, `hsla(${cherenkovHue}, ${sat * 0.15}%, 10%, 0)`);
        haze1.addColorStop(0.6, `hsla(${cherenkovHue}, ${sat * 0.25}%, 8%, ${0.02 + harmonic * 0.03})`);
        haze1.addColorStop(1, `hsla(${cherenkovHue}, ${sat * 0.35}%, 6%, ${0.06 + harmonic * 0.07})`);
        ctx.beginPath();
        ctx.arc(0, 0, outerR, 0, Math.PI * 2);
        ctx.fillStyle = haze1;
        ctx.fill();

        // Cherenkov rim
        const haze2 = ctx.createRadialGradient(0, 0, outerR * 0.88, 0, 0, outerR * 1.05);
        haze2.addColorStop(0, `hsla(${cherenkovHue}, ${sat * 0.5}%, 35%, 0)`);
        haze2.addColorStop(0.5, `hsla(${cherenkovHue}, ${sat * 0.6}%, 40%, ${0.015 + energy * 0.025})`);
        haze2.addColorStop(1, `hsla(${cherenkovHue}, ${sat * 0.4}%, 25%, 0)`);
        ctx.beginPath();
        ctx.arc(0, 0, outerR * 1.05, 0, Math.PI * 2);
        ctx.fillStyle = haze2;
        ctx.fill();

        ctx.restore();
    }

    /**
     * Fractal style — Dense, full-screen psychedelic kaleidoscope.
     *
     * Inspired by tie-dye / crystal macro photography kaleidoscopes:
     * every pixel is covered with saturated color, no black showing.
     *
     * Architecture:
     *   - `screen` blending for colour layers (preserves saturation)
     *   - `lighter` only for sparkle highlights
     *   - 5 recursive depth bands from canvas edge → centre
     *   - Each band counter-rotates with prismatic hue shift
     *   - 3 density layers per band: colour field, texture, sparkle
     *   - Trail persistence accumulates to full opacity over ~1 second
     */
    renderFractalStyle(ctx, centerX, centerY, radius, numSides, hue, thickness) {
        const config = this.config;
        const seed = config.shapeSeed;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const mirrors = config.mirrors;
        const rot = this.accumulatedRotation;
        const sat = config.saturation;

        // Full-field: use canvas diagonal so pattern extends past all edges
        const maxDim = Math.max(this.canvas.width, this.canvas.height) * 0.78;

        const wedgeAngle = Math.PI * 2 / mirrors;
        const halfWedge = wedgeAngle * 0.5;

        ctx.save();
        ctx.translate(centerX, centerY);
        const prevComposite = ctx.globalCompositeOperation;

        // --- Mirror helpers ---
        const mirrorFill = (r, theta, size, color, bandRot) => {
            for (let m = 0; m < mirrors; m++) {
                const base = wedgeAngle * m + bandRot;
                for (let flip = 0; flip < 2; flip++) {
                    const a = base + (flip === 0 ? theta : -theta);
                    ctx.beginPath();
                    ctx.arc(Math.cos(a) * r, Math.sin(a) * r,
                        Math.max(0.5, size), 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        };

        const mirrorGrad = (r, theta, size, c1, c2, bandRot) => {
            for (let m = 0; m < mirrors; m++) {
                const base = wedgeAngle * m + bandRot;
                for (let flip = 0; flip < 2; flip++) {
                    const a = base + (flip === 0 ? theta : -theta);
                    const x = Math.cos(a) * r;
                    const y = Math.sin(a) * r;
                    const g = ctx.createRadialGradient(x, y, 0, x, y, size);
                    g.addColorStop(0, c1);
                    g.addColorStop(0.55, c2);
                    g.addColorStop(1, 'hsla(0,0%,0%,0)');
                    ctx.beginPath();
                    ctx.arc(x, y, size, 0, Math.PI * 2);
                    ctx.fillStyle = g;
                    ctx.fill();
                }
            }
        };

        // ==============================================================
        // RECURSIVE DEPTH BANDS — 5 rings, outer → centre
        // ==============================================================
        const bandCount = 5;
        const scaleF = 0.52;
        const rotSpeeds = [0.2, -0.38, 0.6, -0.85, 1.3];
        const hueShifts = [0, 35, 75, 120, 170];

        // ============= PASS 1: SCREEN blending — saturated colour =====
        ctx.globalCompositeOperation = 'screen';

        for (let band = 0; band < bandCount; band++) {
            const bScale = Math.pow(scaleF, band);
            const fieldR = maxDim * bScale;
            const innerR = band < bandCount - 1
                ? maxDim * Math.pow(scaleF, band + 1) : 0;
            const bRot = rot * rotSpeeds[band];
            const bHue = (hue + hueShifts[band]) % 360;
            const area = bScale * bScale;
            const bSeed = seed + band * 10000;
            // Inner bands are more vivid
            const aBoost = 1.0 + band * 0.25;
            const range = fieldR - innerR;

            // ------ COLOUR FIELD: Large overlapping gradient blobs ------
            // These create the continuous painted base.  Enough overlap
            // that every point in the wedge is hit by at least 2-3 blobs.
            const washN = Math.max(5, Math.round((14 + harmonic * 5) * area));
            for (let i = 0; i < washN; i++) {
                const ws = bSeed + i * 97;
                const bR = innerR + this.seededRandom(ws) * range * 0.92 + range * 0.04;
                const bT = this.seededRandom(ws + 1) * halfWedge;
                const drift = Math.sin(rot * 0.25 + i * 1.3 + band * 2.1) * range * 0.04;
                const r = Math.max(innerR, bR + drift);
                const theta = bT + Math.sin(rot * 0.18 + i * 0.7) * 0.04;
                // BIG blobs — fieldR * 0.12-0.28
                const sz = fieldR * (0.13 + this.seededRandom(ws + 2) * 0.16)
                    * (1 + energy * 0.25);
                const wH = (bHue + 180 + this.seededRandom(ws + 3) * 120) % 360;
                const wL = 22 + this.seededRandom(ws + 4) * 20 + energy * 8;
                const wA = (0.045 + energy * 0.04 + harmonic * 0.025) * aBoost;

                mirrorGrad(r, theta, sz,
                    `hsla(${wH}, ${Math.min(100, sat * 0.85)}%, ${wL}%, ${wA})`,
                    `hsla(${wH}, ${sat * 0.55}%, ${wL * 0.65}%, ${wA * 0.4})`,
                    bRot
                );
            }

            // ------ TEXTURE: Medium noise-displaced blobs ------
            // Organic, irregular shapes that add colour variation.
            const blobN = Math.max(6, Math.round((28 + energy * 14) * area));
            for (let i = 0; i < blobN; i++) {
                const bs = bSeed + 500 + i * 73;
                const baseR = innerR + this.seededRandom(bs) * range * 0.85 + range * 0.05;
                const baseT = this.seededRandom(bs + 1) * halfWedge;
                const bx = baseR * Math.cos(baseT);
                const by = baseR * Math.sin(baseT);
                const nx = this._fractalNoise(bx * 0.006 + rot * 0.12, by * 0.006 + band * 3.7);
                const ny = this._fractalNoise(bx * 0.006 + band * 3.7, by * 0.006 + rot * 0.12);
                const r = Math.max(innerR, baseR + (nx - 0.5) * range * 0.1);
                const theta = Math.max(0, Math.min(halfWedge,
                    baseT + (ny - 0.5) * 0.1));
                // Medium blobs — fieldR * 0.025-0.06
                const sz = fieldR * (0.025 + this.seededRandom(bs + 2) * 0.04)
                    * (1 + energy * 0.5);
                const blobH = (bHue + this.seededRandom(bs + 3) * 90) % 360;
                const blobL = 40 + brightness * 20 + energy * 10;
                const blobA = (0.08 + energy * 0.08 + harmonic * 0.04) * aBoost;

                ctx.fillStyle = `hsla(${blobH}, ${Math.min(100, sat * 1.15)}%, ${blobL}%, ${blobA})`;
                mirrorFill(r, theta, sz, null, bRot);
            }

            // ------ TENDRILS: Thick noise-following strokes ------
            if (band < bandCount - 1) {
                const tendN = Math.max(3, Math.round(
                    (7 + harmonic * 3) * Math.sqrt(area)));
                for (let i = 0; i < tendN; i++) {
                    const ts = bSeed + 2000 + i * 53;
                    const steps = 14 + Math.floor(energy * 7);
                    const pts = [];
                    let cr = innerR + this.seededRandom(ts) * range * 0.6 + range * 0.15;
                    let ct = this.seededRandom(ts + 1) * halfWedge;
                    for (let s = 0; s <= steps; s++) {
                        const fnx = this._fractalNoise(
                            cr * 0.004 + rot * 0.1, ct * 10 + i + band * 5);
                        const fny = this._fractalNoise(
                            ct * 10 + i + band * 5, cr * 0.004 + rot * 0.1);
                        cr = Math.max(innerR, Math.min(fieldR,
                            cr + (fnx - 0.5) * range * 0.06));
                        ct = Math.max(0, Math.min(halfWedge,
                            ct + (fny - 0.5) * 0.035));
                        pts.push({ r: cr, theta: ct });
                    }
                    const tH = (bHue + 100 + this.seededRandom(ts + 2) * 90) % 360;
                    const tA = (0.06 + harmonic * 0.06 + energy * 0.05) * aBoost;
                    const tW = thickness * (0.3 + energy * 0.5)
                        * (0.5 + bScale * 0.5);

                    for (let m = 0; m < mirrors; m++) {
                        const base = wedgeAngle * m + bRot;
                        for (let flip = 0; flip < 2; flip++) {
                            ctx.beginPath();
                            for (let s = 0; s < pts.length; s++) {
                                const p = pts[s];
                                const a = base + (flip === 0 ? p.theta : -p.theta);
                                const px = Math.cos(a) * p.r;
                                const py = Math.sin(a) * p.r;
                                s === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
                            }
                            ctx.strokeStyle = `hsla(${tH}, ${sat * 0.8}%, ${48 + energy * 18}%, ${tA})`;
                            ctx.lineWidth = tW;
                            ctx.lineCap = 'round';
                            ctx.lineJoin = 'round';
                            ctx.stroke();
                        }
                    }
                }
            }

            // ------ BOUNDARY RING between recursive bands ------
            if (band > 0) {
                const rH = (bHue + 60) % 360;
                const rA = 0.06 + energy * 0.08 + harmonic * 0.04;
                ctx.beginPath();
                ctx.arc(0, 0, fieldR, 0, Math.PI * 2);
                ctx.strokeStyle = `hsla(${rH}, ${sat * 0.7}%, 58%, ${rA})`;
                ctx.lineWidth = 0.8 + energy * 1.0;
                ctx.stroke();
            }
        }

        // ============= PASS 2: LIGHTER blending — sparkle/highlights ==
        ctx.globalCompositeOperation = 'lighter';

        for (let band = 0; band < bandCount; band++) {
            const bScale = Math.pow(scaleF, band);
            const fieldR = maxDim * bScale;
            const innerR = band < bandCount - 1
                ? maxDim * Math.pow(scaleF, band + 1) : 0;
            const bRot = rot * rotSpeeds[band];
            const bHue = (hue + hueShifts[band]) % 360;
            const area = bScale * bScale;
            const bSeed = seed + band * 10000;
            const aBoost = 1.0 + band * 0.3;
            const range = fieldR - innerR;

            // ------ STARDUST: Dense bright specks ------
            const dustN = Math.max(10, Math.round((55 + energy * 40) * area));
            for (let i = 0; i < dustN; i++) {
                const ds = bSeed + 1000 + i * 37;
                const r = innerR + this.seededRandom(ds) * range;
                const theta = this.seededRandom(ds + 1) * halfWedge;
                const sz = (0.6 + this.seededRandom(ds + 2) * 3.0
                    * (1 + energy * 1.5)) * (0.5 + bScale * 0.5);
                const dH = (bHue + 30 + this.seededRandom(ds + 3) * 90) % 360;
                const dL = 60 + energy * 25 + brightness * 12;
                const dA = (0.12 + energy * 0.2 + brightness * 0.1) * aBoost;

                ctx.fillStyle = `hsla(${dH}, ${Math.min(100, sat * 1.3)}%, ${dL}%, ${dA})`;
                mirrorFill(r, theta, sz, null, bRot);
            }
        }

        // Restore blending
        ctx.globalCompositeOperation = prevComposite;

        // ------ CENTRAL CORE: Bright warm prismatic glow ------
        const innermost = maxDim * Math.pow(scaleF, bandCount);
        const coreR = innermost * 2.0 * (0.8 + energy * 0.6);
        const coreGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, coreR);
        const coreHue1 = (hue + 170 + rot * 20) % 360; // cyan-ish
        const coreHue2 = (hue + 50) % 360;              // gold-ish
        coreGrad.addColorStop(0, `hsla(${coreHue1}, ${Math.min(100, sat)}%, 88%, ${0.6 + energy * 0.4})`);
        coreGrad.addColorStop(0.25, `hsla(${coreHue2}, ${sat}%, 70%, ${0.35 + energy * 0.3})`);
        coreGrad.addColorStop(0.6, `hsla(${(hue + 240) % 360}, ${sat * 0.7}%, 40%, ${0.1 + energy * 0.1})`);
        coreGrad.addColorStop(1, 'hsla(0, 0%, 0%, 0)');
        ctx.beginPath();
        ctx.arc(0, 0, coreR, 0, Math.PI * 2);
        ctx.fillStyle = coreGrad;
        ctx.fill();

        ctx.restore();
    }

    /**
     * Fractal background — Dense recursive colour field with polar mirroring.
     *
     * Establishes the saturated base that ensures no black pixels remain.
     * Uses `screen` blending for colour richness.  4 recursive bands
     * with large overlapping nebulae, structural arcs, and dense motes.
     */
    renderFractalBackground(ctx, width, height, centerX, centerY, reactivity) {
        const config = this.config;
        const energy = this.smoothedValues.percussiveImpact;
        const harmonic = this.smoothedValues.harmonicEnergy;
        const brightness = this.smoothedValues.spectralBrightness;
        const maxDim = Math.max(width, height) * 0.9;
        const rot = this._bgFractalRotation;
        const accentHsl = this.hexToHsl(config.accentColor);
        const bgHue = accentHsl.h;
        const mirrors = config.mirrors;
        const sat = config.saturation;

        const wedgeAngle = Math.PI * 2 / mirrors;
        const halfWedge = wedgeAngle * 0.5;

        ctx.save();
        ctx.translate(centerX, centerY);

        const prevComposite = ctx.globalCompositeOperation;
        ctx.globalCompositeOperation = 'screen';

        // 4 recursive background depth bands
        const bandCount = 4;
        const scaleF = 0.48;
        const rotSpeeds = [0.1, -0.22, 0.38, -0.55];
        const hueShifts = [0, 45, 95, 145];

        for (let band = 0; band < bandCount; band++) {
            const bScale = Math.pow(scaleF, band);
            const fieldR = maxDim * bScale;
            const innerR = band < bandCount - 1
                ? maxDim * Math.pow(scaleF, band + 1) : 0;
            const bRot = rot * rotSpeeds[band];
            const bHue = (bgHue + hueShifts[band]) % 360;
            const area = bScale * bScale;
            const range = fieldR - innerR;
            const aBoost = 1.0 + band * 0.2;

            // --- Large nebula washes ---
            const nebN = Math.max(4, Math.round(
                (10 + harmonic * reactivity * 4) * area));
            for (let i = 0; i < nebN; i++) {
                const ns = 7777 + band * 3000 + i * 131;
                const r = innerR + this.seededRandom(ns) * range * 0.85 + range * 0.08;
                const theta = this.seededRandom(ns + 1) * halfWedge;
                const drift = Math.sin(rot * 0.12 + i * 1.7 + band * 2.3) * range * 0.03;
                const nr = r + drift;
                // BIG nebulae — fieldR * 0.12-0.28
                const sz = fieldR * (0.13 + this.seededRandom(ns + 2) * 0.16)
                    * (1 + energy * reactivity * 0.15);
                const nH = (bHue + 180 + this.seededRandom(ns + 3) * 100) % 360;
                const nL = 15 + this.seededRandom(ns + 4) * 16;
                const nA = (0.03 + energy * reactivity * 0.025 + harmonic * 0.015) * aBoost;

                for (let m = 0; m < mirrors; m++) {
                    const base = wedgeAngle * m + bRot;
                    for (let flip = 0; flip < 2; flip++) {
                        const a = base + (flip === 0 ? theta : -theta);
                        const x = Math.cos(a) * nr;
                        const y = Math.sin(a) * nr;
                        const g = ctx.createRadialGradient(x, y, 0, x, y, sz);
                        g.addColorStop(0, `hsla(${nH}, ${sat * 0.55}%, ${nL}%, ${nA})`);
                        g.addColorStop(0.5, `hsla(${nH}, ${sat * 0.35}%, ${nL * 0.6}%, ${nA * 0.4})`);
                        g.addColorStop(1, 'hsla(0,0%,0%,0)');
                        ctx.beginPath();
                        ctx.arc(x, y, sz, 0, Math.PI * 2);
                        ctx.fillStyle = g;
                        ctx.fill();
                    }
                }
            }

            // --- Structural noise-contour arcs ---
            const arcN = Math.max(3, Math.round(
                (6 + energy * reactivity * 3) * Math.sqrt(area)));
            for (let i = 0; i < arcN; i++) {
                const as = 5555 + band * 3000 + i * 89;
                const arcSteps = 18;
                const startR = innerR + this.seededRandom(as) * range * 0.6 + range * 0.15;
                const startT = this.seededRandom(as + 1) * halfWedge;
                const pts = [];
                let cr = startR, ct = startT;
                for (let s = 0; s <= arcSteps; s++) {
                    const fnx = this._fractalNoise(
                        cr * 0.003 + rot * 0.07, ct * 8 + i + band * 7);
                    const fny = this._fractalNoise(
                        ct * 8 + i + band * 7, cr * 0.003 + rot * 0.07);
                    cr = Math.max(innerR, Math.min(fieldR,
                        cr + (fnx - 0.5) * range * 0.05));
                    ct = Math.max(0, Math.min(halfWedge,
                        ct + (fny - 0.5) * 0.025));
                    pts.push({ r: cr, theta: ct });
                }

                const aH = (bHue + 130 + this.seededRandom(as + 2) * 70) % 360;
                const aA = (0.035 + harmonic * reactivity * 0.04) * aBoost;

                for (let m = 0; m < mirrors; m++) {
                    const base = wedgeAngle * m + bRot;
                    for (let flip = 0; flip < 2; flip++) {
                        ctx.beginPath();
                        for (let s = 0; s < pts.length; s++) {
                            const p = pts[s];
                            const pa = base + (flip === 0 ? p.theta : -p.theta);
                            const x = Math.cos(pa) * p.r;
                            const y = Math.sin(pa) * p.r;
                            s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                        }
                        ctx.strokeStyle = `hsla(${aH}, ${sat * 0.45}%, 42%, ${aA})`;
                        ctx.lineWidth = (1.0 + energy * reactivity * 1.2)
                            * (0.5 + bScale * 0.5);
                        ctx.lineCap = 'round';
                        ctx.stroke();
                    }
                }
            }

            // --- Dense texture motes ---
            const moteN = Math.max(6, Math.round(
                (18 + energy * reactivity * 10) * area));
            for (let i = 0; i < moteN; i++) {
                const ms = 3333 + band * 3000 + i * 47;
                const r = innerR + this.seededRandom(ms) * range;
                const theta = this.seededRandom(ms + 1) * halfWedge;
                const sz = (0.5 + this.seededRandom(ms + 2) * 2.5
                    * (1 + energy * reactivity)) * (0.5 + bScale * 0.5);
                const mH = (bHue + this.seededRandom(ms + 3) * 130) % 360;
                const mA = (0.04 + energy * reactivity * 0.06 + brightness * 0.02) * aBoost;

                for (let m = 0; m < mirrors; m++) {
                    const base = wedgeAngle * m + bRot;
                    for (let flip = 0; flip < 2; flip++) {
                        const ma = base + (flip === 0 ? theta : -theta);
                        const x = Math.cos(ma) * r;
                        const y = Math.sin(ma) * r;
                        ctx.beginPath();
                        ctx.arc(x, y, sz, 0, Math.PI * 2);
                        ctx.fillStyle = `hsla(${mH}, ${sat * 0.5}%, 48%, ${mA})`;
                        ctx.fill();
                    }
                }
            }

            // Band boundary ring
            if (band > 0) {
                const rH = (bHue + 90) % 360;
                const rA = 0.03 + energy * reactivity * 0.04;
                ctx.beginPath();
                ctx.arc(0, 0, fieldR, 0, Math.PI * 2);
                ctx.strokeStyle = `hsla(${rH}, ${sat * 0.4}%, 42%, ${rA})`;
                ctx.lineWidth = 0.5 + energy * reactivity * 0.4;
                ctx.stroke();
            }
        }

        ctx.globalCompositeOperation = prevComposite;
        ctx.restore();
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
    }

    return ChromascopeRenderer;
}));
