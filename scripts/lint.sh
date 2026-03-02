#!/bin/bash
# scripts/lint.sh — Run linters

set -e

echo "🔍 Running linters..."
ruff check src/
black --check src/

echo "✅ Linting passed!"
