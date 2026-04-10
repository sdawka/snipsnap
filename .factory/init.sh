#!/bin/bash
set -e

cd /Users/sdawka/Code/snipsnap

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate venv and install dependencies
source .venv/bin/activate
pip install -e ".[dev]" --quiet 2>/dev/null || true

# Create .env from example if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "Created .env from .env.example — fill in your API keys"
fi

# Create data directory
mkdir -p snipsnap_data

echo "SnipSnap environment ready"
