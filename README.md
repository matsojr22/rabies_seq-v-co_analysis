# Rabies Sequence vs Co Analysis

Comprehensive statistical analysis script for rabies tracing data comparison between injection schemes with advanced stratified analytics and convergence index calculations.

## Features

### Core Analysis
- **Statistical comparisons** between injection schemes (coinjected vs separated)
- **Multiple normalization methods** (area-normalized, ratio-based, proportion-based)
- **Power analysis** with effect size calculations
- **Practical effect thresholds** for experimental design

### Advanced Analytics
- **Convergence Index Analysis**:
  - Standard convergence index (V1 layers only)
  - Convergence index ALL (all input cells)
  - Convergence index V1 (V1 layers only)
- **Starter Cell Correlation Analysis**:
  - Raw count correlations
  - Area-normalized correlations
  - Labeling efficiency analysis
  - Preference ratio analysis
- **All V1 Layers Analysis**:
  - Individual layer correlations
  - Normalized layer analysis
  - Efficiency-based layer analysis
- **L4/L5 Preference Analysis**:
  - Enhanced statistical testing with Bonferroni corrections
  - Multiple preference metrics
  - Confounding variable analysis
- **Stratified Analytics**:
  - Quintile-based stratification
  - ANCOVA analysis
  - Residual analysis
  - Starter cell density analysis
  - Efficiency-based stratification

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
# Run the complete analysis
python rabies_compare_2.py

# The script will generate:
# - Statistical analysis CSV files
# - PNG and SVG plots
# - Power analysis summaries
# - Practical effect threshold reports
# - Convergence index analyses
# - Stratified analytics
# - Enhanced preference analysis
```

## Output Files

### Statistical Results
- **`rabies_analysis_stats_*.csv`** - Statistical test results
- **`rabies_analysis_summary_*.csv`** - Summary statistics

### Plots and Visualizations
- **`rabies_analysis_*.png`** and **`rabies_analysis_*.svg`** - Analysis plots
- **`convergence_indices*.png`** - Convergence index plots
- **`stratified_analytics/`** - Stratified analysis outputs

### Analysis Reports
- **`rabies_analysis_power_analysis_summary.xlsx`** - Power analysis
- **`rabies_analysis_practical_effect_summary.xlsx`** - Effect thresholds
- **`rabies_analysis_l4_l5_preference_interpretation_enhanced.txt`** - Detailed interpretation

### Convergence Index Files
- **`convergence_indices.csv`** - Standard convergence index
- **`convergence_indices_ALL.csv`** - All input cells convergence
- **`convergence_indices_V1.csv`** - V1 layers convergence

### Stratified Analytics
- **`stratified_analytics/stratified_analysis_summary.txt`** - Comprehensive summary
- **`stratified_analytics/*.png`** - Stratified analysis plots
- **`stratified_analytics/*.csv`** - Stratified analysis results

## Data Format

Input data should be in CSV format with columns:
- **Animal ID** - Unique identifier for each animal
- **Injection Scheme** - Either "coinjected" or "separated"
- **V1 Starters** - Number of starter cells in V1
- **V1 Area (pixels)** - Area of V1 region in pixels
- **Brain region cell counts** - Counts for each brain region and layer
  - V1 L2/3, V1 L4, V1 L5, V1 L6a, V1 L6b
  - V2M, V2L, dLGN

See `data/rabies_comparison.csv` for example format.

## Key Analysis Components

### 1. Convergence Index Analysis
- **Standard**: V1 layers / Starter cells
- **ALL**: All input cells / Starter cells  
- **V1**: V1 layers only / Starter cells
- Individual layer convergence indices

### 2. Starter Cell Correlation Analysis
- Correlation between starter cell numbers and input patterns
- Area-normalized correlations
- Labeling efficiency calculations
- Preference ratio analysis

### 3. Stratified Analytics
- Addresses confounding variables (especially starter cell number differences)
- Multiple analytical approaches to test injection scheme effects
- ANCOVA, residual analysis, and efficiency-based stratification

### 4. Enhanced L4/L5 Preference Analysis
- Multiple preference metrics
- Bonferroni corrections for multiple testing
- Confounding variable assessment
- Comprehensive statistical interpretation

## Statistical Methods

- **Normality testing**: Shapiro-Wilk test
- **Variance equality**: Levene's test
- **Group comparisons**: Student's t-test, Welch's t-test, Mann-Whitney U test
- **Effect sizes**: Cohen's d, Hedges' g
- **Power analysis**: T-test power calculations
- **Multiple testing**: Bonferroni corrections
- **Stratified analysis**: ANCOVA, residual analysis, quintile stratification

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure all dependencies are installed
2. **Memory issues**: Large datasets may require increased memory allocation
3. **Plot generation**: Ensure matplotlib backend is properly configured

### Dependencies
- Core: pandas, numpy, scipy, statsmodels
- ML: scikit-learn (for stratified analytics)
- Visualization: matplotlib, seaborn
- Excel: openpyxl
- Optional: plotly (for enhanced plots)

## Citation

If you use this analysis script in your research, please cite appropriately and acknowledge the statistical methods used.
