#!/bin/bash
# Script to run patch generation in background with nohup

echo "🚀 Starting comprehensive patch generation with nohup..."
echo "📝 Logs will be written to: patch_generation.log and nohup.out"
echo "🔍 Monitor progress with: tail -f patch_generation.log"
echo ""

# Run with nohup - redirect stdout to nohup.out, stderr to patch_generation.log
nohup poetry run python generate_all_patches.py "$@" > nohup.out 2>&1 &

# Get the process ID
PID=$!

echo "✅ Patch generation started in background!"
echo "   Process ID: $PID"
echo "   Monitor logs: tail -f patch_generation.log"
echo "   Monitor output: tail -f nohup.out"
echo "   Check process: ps -p $PID"
echo "   Kill if needed: kill $PID"
echo ""
echo "🎯 Expected to generate patches for all modes (train/validation/test)"
echo "⏱️  This may take several hours depending on data size"
