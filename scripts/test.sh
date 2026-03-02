#!/bin/bash
# scripts/test.sh — Run all tests

set -e

echo "🧪 Running tests..."
pytest tests/ -v --cov=kokao --cov-report=term-missing

echo "✅ All tests passed!"
