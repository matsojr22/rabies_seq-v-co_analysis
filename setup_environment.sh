#!/bin/bash

# Setup script for rabies analysis environment
echo "Setting up rabies analysis environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda detected. Creating environment from environment.yml..."
    conda env create -f environment.yml
    echo ""
    echo "To activate the environment:"
    echo "conda activate rabies_analysis"
else
    echo "Conda not found. Installing packages with pip..."
    pip install -r requirements.txt
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
import sklearn
import openpyxl
print('✓ All required packages are installed successfully!')
"

echo ""
echo "Setup complete!"
echo ""
echo "To run the analysis:"
echo "python rabies_compare_2.py"
echo ""
echo "For detailed usage instructions, see README.md"
