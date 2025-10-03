# Rabies Sequence vs Co Analysis

Statistical analysis script for rabies tracing data comparison between injection schemes.

## Requirements

- Python 3.11+
- Required packages listed in `requirements.txt` and `environment.yml`

## Installation

### Option 1: Using Conda (Recommended)
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate rabies_analysis
```

### Option 2: Using Pip
```bash
# Install packages
pip install -r requirements.txt
```

### Option 3: Automated Setup
```bash
# Run setup script
./setup_environment.sh
```

## Usage

```bash
# Run the analysis
python rabies_compare_2.py

# The script will generate:
# - Statistical analysis CSV files
# - PNG and SVG plots
# - Power analysis summaries
# - Practical effect threshold reports
```

## Output Files

- **Statistical Results**: `rabies_analysis_stats_*.csv`
- **Plots**: `rabies_analysis_*.png` and `rabies_analysis_*.svg`
- **Power Analysis**: `rabies_analysis_power_analysis_summary.xlsx`
- **Practical Thresholds**: `rabies_analysis_practical_effect_summary.xlsx`

## Data Format

Input data should be in CSV format with columns:
- Animal ID
- Injection Scheme
- Brain region cell counts
- Starter cell counts
- Area measurements

See `data/rabies_comparison.csv` for example format.
