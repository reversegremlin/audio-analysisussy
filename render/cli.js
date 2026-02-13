#!/usr/bin/env node
/**
 * Chromascope Server-Side Renderer
 *
 * Reads a frame manifest (from Python audio analysis) and a config JSON,
 * renders each frame via the shared ChromascopeRenderer (node-canvas),
 * and pipes raw RGBA pixels to ffmpeg which produces an MP4.
 *
 * Usage:
 *   node cli.js --manifest manifest.json --config config.json \
 *               --audio input.mp3 --output output.mp4 \
 *               [--width 1920] [--height 1080] [--fps 60]
 *
 * Progress is reported as JSON lines on stderr:
 *   {"type":"progress","frame":N,"total":T,"percent":P}
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { createCanvas } = require('canvas');

// Load the shared render engine (UMD module)
const ChromascopeRenderer = require(path.join(__dirname, '..', 'frontend', 'render-engine.js'));

// ---------------------------------------------------------------------------
// Encoding quality profiles & argument parsing
// ---------------------------------------------------------------------------
const QUALITY_PROFILES = {
    high: {
        preset: 'slow',
        crf: '16',
        pixFmt: 'yuv444p',
        profile: 'high444',
        tune: 'animation',
        audioBitrate: '256k',
    },
    medium: {
        preset: 'medium',
        crf: '18',
        pixFmt: 'yuv420p',
        profile: 'high',
        tune: 'animation',
        audioBitrate: '192k',
    },
    fast: {
        preset: 'fast',
        crf: '22',
        pixFmt: 'yuv420p',
        profile: 'high',
        tune: null,
        audioBitrate: '128k',
    },
};

function parseArgs() {
    const args = process.argv.slice(2);
    const opts = {
        manifest: null,
        config: null,
        audio: null,
        output: null,
        width: 1920,
        height: 1080,
        fps: 60,
        quality: 'high',
    };
    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--manifest': opts.manifest = args[++i]; break;
            case '--config':   opts.config   = args[++i]; break;
            case '--audio':    opts.audio    = args[++i]; break;
            case '--output':   opts.output   = args[++i]; break;
            case '--width':    opts.width    = parseInt(args[++i], 10); break;
            case '--height':   opts.height   = parseInt(args[++i], 10); break;
            case '--fps':      opts.fps      = parseInt(args[++i], 10); break;
            case '--quality':  opts.quality  = args[++i]; break;
        }
    }

    if (!opts.manifest || !opts.config || !opts.audio || !opts.output) {
        process.stderr.write(
            'Usage: node cli.js --manifest FILE --config FILE --audio FILE --output FILE\n' +
            '       [--width 1920] [--height 1080] [--fps 60] [--quality high|medium|fast]\n'
        );
        process.exit(1);
    }

    if (!QUALITY_PROFILES[opts.quality]) {
        process.stderr.write(
            `Unknown quality "${opts.quality}". Choose from: ${Object.keys(QUALITY_PROFILES).join(', ')}\n`
        );
        process.exit(1);
    }

    return opts;
}

// ---------------------------------------------------------------------------
// Progress reporting (JSON lines on stderr)
// ---------------------------------------------------------------------------
function reportProgress(frame, total, message) {
    const percent = Math.round((frame / total) * 100);
    process.stderr.write(
        JSON.stringify({ type: 'progress', frame, total, percent, message }) + '\n'
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
    const opts = parseArgs();

    // Load manifest and config
    const manifest = JSON.parse(fs.readFileSync(opts.manifest, 'utf8'));
    const config = JSON.parse(fs.readFileSync(opts.config, 'utf8'));

    // Override resolution / fps from CLI flags
    config.width = opts.width;
    config.height = opts.height;
    config.fps = opts.fps;

    const frames = manifest.frames;
    const totalFrames = frames.length;
    const deltaTime = 1000 / opts.fps; // ms per frame

    reportProgress(0, totalFrames, 'Initializing renderer...');

    // Create a persistent node-canvas (trail effect relies on previous frame content)
    const canvas = createCanvas(opts.width, opts.height);
    const renderer = new ChromascopeRenderer(config, canvas);

    // Spawn ffmpeg — raw RGBA on stdin, MP4 on disk
    const qp = QUALITY_PROFILES[opts.quality];
    const ffmpegArgs = [
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgba',
        '-s', `${opts.width}x${opts.height}`,
        '-r', String(opts.fps),
        '-i', 'pipe:0',
        '-i', opts.audio,
        '-c:v', 'libx264',
        '-profile:v', qp.profile,
        '-preset', qp.preset,
        '-crf', qp.crf,
        ...(qp.tune ? ['-tune', qp.tune] : []),
        '-pix_fmt', qp.pixFmt,
        // Tag color space so players interpret colors correctly
        '-colorspace', 'bt709',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-c:a', 'aac',
        '-b:a', qp.audioBitrate,
        '-movflags', '+faststart',
        '-shortest',
        opts.output,
    ];

    const ffmpeg = spawn('ffmpeg', ffmpegArgs, {
        stdio: ['pipe', 'pipe', 'pipe'],
    });

    // Collect ffmpeg stderr for error reporting
    let ffmpegStderr = '';
    ffmpeg.stderr.on('data', (chunk) => { ffmpegStderr += chunk.toString(); });

    const ffmpegDone = new Promise((resolve, reject) => {
        ffmpeg.on('close', (code) => {
            if (code === 0) resolve();
            else reject(new Error(`ffmpeg exited with code ${code}: ${ffmpegStderr.slice(-500)}`));
        });
        ffmpeg.on('error', reject);
    });

    // Helper: write buffer with backpressure handling
    function writeFrame(buf) {
        return new Promise((resolve, reject) => {
            const ok = ffmpeg.stdin.write(buf);
            if (ok) {
                resolve();
            } else {
                ffmpeg.stdin.once('drain', resolve);
                ffmpeg.stdin.once('error', reject);
            }
        });
    }

    // Render loop
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    const reportInterval = Math.max(1, Math.floor(totalFrames / 100)); // ~1% steps

    for (let i = 0; i < totalFrames; i++) {
        const frameData = frames[i];

        // Render frame (isPlaying = true for export)
        renderer.renderFrame(frameData, deltaTime, true);

        // Extract raw RGBA pixels
        const imageData = ctx.getImageData(0, 0, opts.width, opts.height);
        await writeFrame(Buffer.from(imageData.data.buffer));

        // Report progress periodically
        if (i % reportInterval === 0 || i === totalFrames - 1) {
            reportProgress(i + 1, totalFrames, `Rendering frame ${i + 1}/${totalFrames}`);
        }
    }

    // Close ffmpeg stdin and wait for it to finish
    ffmpeg.stdin.end();
    reportProgress(totalFrames, totalFrames, 'Encoding final MP4...');
    await ffmpegDone;

    reportProgress(totalFrames, totalFrames, 'Done');
}

main().catch((err) => {
    process.stderr.write(JSON.stringify({ type: 'error', message: err.message }) + '\n');
    process.exit(1);
});
