#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
# Check OS to determine activation script location
if [ -d "venv/Scripts" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install ipykernel
echo "Installing ipykernel..."
pip install ipykernel

# Add to Jupyter
echo "Registering kernel..."
python3 -m ipykernel install --user --name=rag-minimal --display-name "Python (RAG Minimal)"

echo "Setup complete! You can now select 'Python (RAG Minimal)' as your kernel in Jupyter."
