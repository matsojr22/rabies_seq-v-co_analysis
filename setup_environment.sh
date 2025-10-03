#!/bin/bash

# Setup script for rabies analysis environment
echo "Setting up rabies analysis environment..."

# Option 1: Create conda environment from yml file
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Option 2: Install packages in existing environment
echo "Installing packages with pip..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "conda activate rabies_analysis"
echo ""
echo "To run the analysis:"
echo "python rabies_compare_2.py"
