#!/bin/bash
# scripts/build.sh — Build package

set -e

echo "📦 Building package..."
python -m build --wheel

echo "✅ Build completed!"
