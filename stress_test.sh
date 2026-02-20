#!/bin/bash

# Chromascope Architecture Stress Test
# Generates 15-second clips for each mode using "extreme" architectural settings.

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_audio> <output_base_name>"
    echo "Example: $0 music.mp3 stress_test"
    exit 1
fi

INPUT_AUDIO=$1
OUTPUT_BASE=$2
MODES=("fractal" "solar" "decay" "mixed")

# Ensure we are in the project root and can find the src
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "🚀 Starting Chromascope Stress Test..."
echo "Audio: $INPUT_AUDIO"
echo "------------------------------------"

for MODE in "${MODES[@]}"; do
    OUT_FILE="${OUTPUT_BASE}_${MODE}.mp4"
    echo "🎨 Rendering Mode: $MODE (Extreme Settings)"
    
    # We use 'cycle' for both mirror and interference to stress the dynamic transitions
    # We use 'high' profile to push resolution and complexity
    python3 -m chromascope.experiment.cli "$INPUT_AUDIO" \
        --mode "$MODE" \
        --output "$OUT_FILE" \
        --max-duration 15 \
        --profile high \
        --mirror cycle \
        --interference cycle \
        --palette $([ "$MODE" == "solar" ] && echo "solar" || echo "jewel")

    echo "✅ Finished $MODE -> $OUT_FILE"
    echo "------------------------------------"
done

echo "🎉 Stress Test Complete! All extreme variants generated."
