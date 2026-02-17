# Chromascope Solar Visualizer

This document describes the `chromascope-solar` audio-reactive visualizer, a tool that generates a solar flare/sunspot simulation from an audio file.

## Basic Usage

To generate a video, run the following command:

```bash
chromascope-solar <path_to_audio_file>
```

This will create a video file named `<audio_file_stem>_solar.mp4` in the same directory as the audio file.

## Command-Line Options

You can customize the output with the following options:

| Option | Description | Default |
| --- | --- | --- |
| `audio` | Input audio file (wav, mp3, flac) | (required) |
| `-o`, `--output` | Output MP4 path | `<audio>_solar.mp4` |
| `--width` | Video width | 1920 |
| `--height` | Video height | 1080 |
| `-f`, `--fps` | Frames per second | 60 |
| `--pan-speed-x` | Horizontal pan speed | 0.1 |
| `--pan-speed-y` | Vertical pan speed | 0.05 |
| `--zoom-speed` | Zoom speed | 0.05 |
| `--colormap` | Color map to use (`default`, `blue`, `green`) | `default` |
| `--no-glow` | Disable glow | (disabled) |
| `--no-aberration` | Disable chromatic aberration | (disabled) |
| `--no-vignette` | Disable vignette | (disabled) |
| `--max-duration` | Limit output to N seconds | (full duration) |
| `-q`, `--quality` | Encoding quality (`high`, `medium`, `fast`) | `high` |

## Example Commands

Here are a few examples of how to use the `chromascope-solar` command with different options.

**Generate a short, fast preview:**

```bash
chromascope-solar tests/assets/shineglowglisten.mp3 -q fast --max-duration 10
```

**Generate a high-quality video with the blue colormap:**

```bash
chromascope-solar tests/assets/shineglowglisten.mp3 -q high --colormap blue
```

**Generate a video with faster panning and zooming:**

```bash
chromascope-solar tests/assets/shineglowglisten.mp3 --pan-speed-x 0.2 --zoom-speed 0.1
```

**Generate a 4K video:**

```bash
chromascope-solar tests/assets/shineglowglisten.mp3 --width 3840 --height 2160
```
