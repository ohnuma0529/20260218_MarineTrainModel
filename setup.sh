#!/bin/bash
# setup.sh

# Exit on error
set -e

echo "=== Fish Detection Model Training Environment Setup ==="

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found."
    exit 1
fi

# Install ultralytics (ensure the latest or target version)
# pip install ultralytics

echo "=== Setup Complete ==="
echo "To activate the environment, run: source venv/bin/activate"
