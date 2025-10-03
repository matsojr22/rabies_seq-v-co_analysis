"""
Rabies tracing comparison: coinjection vs sequential injection (Improved Version)
- Normalization: (cells / starter_cells) / area
- Note: 'area' is given in **pixels** and assumed to be a consistent unit across comparisons. For absolute values or unit conversions, rescale externally.
- Graphs: local layers and long-distance regions
- Stats: normality (Shapiro), variance (Levene), appropriate test per stratum (Student/Welch t; Mann–Whitney if non-normal)
- Outputs: figures and a summary CSV written to current directory

Data format: Wide format CSV with columns for each layer/region
- Animal ID: identifier for biological replicate
- Injection Scheme: experimental condition ('coinjected' or 'separated')
- V1 L2/3, V1 L4, V1 L5, V1 L6a, V1 L6b: local layer counts
- V2M, V2L, dLGN: long-distance region counts
- V1 Starters: number of starter cells
- Area columns: area in pixels for each region

IMPROVEMENTS in v2:
- Fixed random jitter with seed for reproducible plots
- Improved SEM calculation with proper single-point handling
- Added non-parametric effect size (r = Z/√N) for Mann-Whitney U tests
- Better documentation of proportion analysis assumptions
- Enhanced error handling and edge case management
"""

from __future__ import annotations
import sys
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import ttest_power, tt_ind_solve_power
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

# Set random seed for reproducible plots
np.random.seed(42)

# -------------------------------
# Config: data transformation and analysis settings
# -------------------------------

# Long-distance regions to analyze separately
LONG_DISTANCE_AREAS = ["V2M", "V2L", "dLGN"]

# Local site regions (V1 layers)
LOCAL_SITE_NAMES = ["V1"]

# Layer mapping for V1 data
LAYER_MAPPING = {
    "V1 L2/3": "L2/3",
    "V1 L4": "L4",
    "V1 L5": "L5",
    "V1 L6a": "L6a",
    "V1 L6b": "L6b"
}

# Area column mapping
AREA_MAPPING = {
    "V1": "V1 Area (pixels)",
    "V2M": "V2M Area (pixels)",
    "V2L": "V2L Area (pixels)", 
    "dLGN": "dLGN Area (pixels)"
}

# Plot settings
PLOT_DPI = 150
FIGSIZE = (6, 4)

# ----------------------------------
# Data transformation functions
# ----------------------------------

def transform_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Transform wide format data to long format for analysis."""
    # Clean the data - remove empty rows
    df = df.dropna(subset=['Animal ID', 'Injection Scheme']).copy()
    
    # Rename columns for consistency
    df = df.rename(columns={
        'Animal ID': 'SampleID',
        'Injection Scheme': 'Condition',
        'V1 Starters': 'Starter'
    })
    
    # Standardize condition names
    df['Condition'] = df['Condition'].str.lower()
    
    long_data = []
    
    # Process V1 layers (local data)
    for layer_col, layer_name in LAYER_MAPPING.items():
        if layer_col in df.columns:
            layer_data = df[['SampleID', 'Condition', 'Starter', layer_col]].copy()
            layer_data['Region'] = 'V1'
            layer_data['Layer'] = layer_name
            layer_data['Cells'] = layer_data[layer_col]
            layer_data['Area'] = df['V1 Area (pixels)']
            long_data.append(layer_data[['SampleID', 'Condition', 'Region', 'Layer', 'Cells', 'Starter', 'Area']])
    
    # Process long-distance regions
    for region in LONG_DISTANCE_AREAS:
        if region in df.columns:
            region_data = df[['SampleID', 'Condition', 'Starter', region]].copy()
            region_data['Region'] = region
            region_data['Layer'] = None  # No layer info for long-distance
            region_data['Cells'] = region_data[region]
            region_data['Area'] = df[AREA_MAPPING[region]]
            long_data.append(region_data[['SampleID', 'Condition', 'Region', 'Layer', 'Cells', 'Starter', 'Area']])
    
    # Combine all data
    result_df = pd.concat(long_data, ignore_index=True)
    
    # Convert numeric columns
    for col in ['Cells', 'Starter', 'Area']:
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    
    return result_df


def assert_two_conditions(df: pd.DataFrame) -> List[str]:
    conds = sorted(df["Condition"].dropna().astype(str).unique())
    if len(conds) != 2:
        raise ValueError(f"Expected exactly 2 conditions, found {len(conds)}: {conds}")
    return conds

# ----------------------
# Normalization & strata
# ----------------------

def compute_normalized(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Cells", "Starter", "Area"]:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    df = df.copy()
    df["Starter"] = df["Starter"].replace({0: np.nan})
    df["Area"] = df["Area"].replace({0: np.nan})
    df["Norm"] = (df["Cells"] / df["Starter"]) / df["Area"]
    return df

def compute_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cells/starter ratio without area normalization."""
    for col in ["Cells", "Starter"]:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    df = df.copy()
    df["Starter"] = df["Starter"].replace({0: np.nan})
    df["Ratio"] = df["Cells"] / df["Starter"]
    return df

def compute_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute proportions where layers sum to 100% and regions sum to 100%.
    
    NOTE: This approach normalizes V1 layers and long-distance regions separately.
    This assumes that:
    1. V1 layers represent the local injection site distribution
    2. Long-distance regions represent the projection distribution
    3. These are biologically distinct compartments that should be analyzed separately
    
    If you need a unified proportion analysis, consider normalizing all regions together.
    """
    df = df.copy()
    
    # Calculate proportions for each animal and condition
    proportions = []
    
    for (animal, condition), group in df.groupby(['SampleID', 'Condition']):
        # Calculate layer proportions (V1 layers sum to 100%)
        layer_data = group[group['Region'] == 'V1'].copy()
        if not layer_data.empty:
            # Handle missing data by filling with 0
            layer_data['Cells'] = layer_data['Cells'].fillna(0)
            layer_total = layer_data['Cells'].sum()
            if layer_total > 0:
                layer_data['Proportion'] = (layer_data['Cells'] / layer_total) * 100
                proportions.append(layer_data)
            else:
                # If total is 0, all layers get 0% proportion
                layer_data['Proportion'] = 0.0
                proportions.append(layer_data)
        
        # Calculate region proportions (long-distance regions sum to 100%)
        region_data = group[group['Region'].isin(LONG_DISTANCE_AREAS)].copy()
        if not region_data.empty:
            # Handle missing data by filling with 0
            region_data['Cells'] = region_data['Cells'].fillna(0)
            region_total = region_data['Cells'].sum()
            if region_total > 0:
                region_data['Proportion'] = (region_data['Cells'] / region_total) * 100
                proportions.append(region_data)
            else:
                # If total is 0, all regions get 0% proportion
                region_data['Proportion'] = 0.0
                proportions.append(region_data)
    
    if proportions:
        result_df = pd.concat(proportions, ignore_index=True)
        return result_df
    else:
        return df

def compute_john_proportions(df_orig: pd.DataFrame, proportion_type='all_non_starter') -> pd.DataFrame:
    """
    Compute proportions using John's method from Rapid Rabies Methods paper.
    
    This mimics the logic from the Rapid Rabies Methods paper.py script.
    
    Parameters:
    df_orig (pd.DataFrame): Original wide-format dataframe
    proportion_type (str): Either 'all_non_starter' or 'local_vs_long_distance'
    
    Returns:
    pd.DataFrame: Long-format dataframe with John's proportion calculations
    """
    df = df_orig.copy()
    
    # Define area columns (matching John's approach)
    v1_areas = ['V1 L2/3', 'V1 L4', 'V1 L5', 'V1 L6a', 'V1 L6b']
    non_v1_areas = ['V2M', 'V2L', 'dLGN']
    all_areas = v1_areas + non_v1_areas
    
    # Create area counts with combined L2/3 (matching John's logic)
    area_counts = df[['V1 L4', 'V1 L5', 'V1 L6a', 'V1 L6b', 'V2M', 'V2L', 'dLGN']].copy()
    
    # Handle different column formats
    if 'V1 L2/3 Upper' in df.columns and 'V1 L2/3 Lower' in df.columns:
        # Original format with separate Upper/Lower columns
        area_counts['V1 L2/3'] = df['V1 L2/3 Upper'] + df['V1 L2/3 Lower']
    elif 'V1 L2/3' in df.columns:
        # Already combined format
        area_counts['V1 L2/3'] = df['V1 L2/3']
    else:
        raise ValueError("Could not find V1 L2/3 columns in expected formats")
    
    # Note: John's script handles starter subtraction based on strain, but since
    # the user has already manually subtracted starters, we use the counts as-is
    
    # Calculate proportions for each injection scheme group
    injection_schemes = df['Injection Scheme'].unique()
    all_proportions = []
    
    for scheme in injection_schemes:
        scheme_mask = df['Injection Scheme'] == scheme
        scheme_area_counts = area_counts[scheme_mask]
        scheme_df = df[scheme_mask]
        
        if proportion_type == 'all_non_starter':
            # Calculate proportions relative to all non-starter inputs
            total_non_starter = scheme_area_counts[all_areas].sum(axis=1)
            proportions = scheme_area_counts[all_areas].div(total_non_starter, axis=0)
            
        elif proportion_type == 'local_vs_long_distance':
            # Calculate local (V1) and long-distance (non-V1) totals
            local_total = scheme_area_counts[v1_areas].sum(axis=1)
            long_distance_total = scheme_area_counts[non_v1_areas].sum(axis=1)
            
            # Initialize proportions dataframe
            proportions = pd.DataFrame(index=scheme_area_counts.index, columns=all_areas)
            
            # Calculate proportions for V1 areas (relative to local inputs)
            for area in v1_areas:
                proportions[area] = scheme_area_counts[area] / local_total
            
            # Calculate proportions for non-V1 areas (relative to long-distance inputs)
            for area in non_v1_areas:
                proportions[area] = scheme_area_counts[area] / long_distance_total
        
        else:
            raise ValueError("proportion_type must be 'all_non_starter' or 'local_vs_long_distance'")
        
        # Handle NaN values
        proportions = proportions.fillna(0)
        
        # Convert to long format for consistency with other functions
        for area in all_areas:
            for i, (idx, row) in enumerate(proportions.iterrows()):
                # Get the corresponding row from the original dataframe using position
                animal_id = scheme_df.iloc[i]['Animal ID']
                condition = scheme_df.iloc[i]['Injection Scheme']
                
                # Determine region and layer
                if area in v1_areas:
                    region = 'V1'
                    layer = area.split(' ')[1]  # Extract L2/3, L4, etc.
                else:
                    region = area
                    layer = None
                
                all_proportions.append({
                    'SampleID': animal_id,
                    'Condition': condition.lower(),
                    'Region': region,
                    'Layer': layer,
                    'Cells': scheme_area_counts.iloc[i][area],
                    'Starter': scheme_df.iloc[i]['V1 Starters'],
                    'Area': scheme_df.iloc[i]['V1 Area (pixels)'] if region == 'V1' else scheme_df.iloc[i][f'{area} Area (pixels)'],
                    'John_Proportion': row[area] * 100  # Convert to percentage
                })
    
    result_df = pd.DataFrame(all_proportions)
    return result_df
# ----------------------
# Statistical comparisons
# ----------------------
@dataclass
class TestResult:
    stratum_type: str
    stratum: str
    n1: int
    n2: int
    normal1: bool
    normal2: bool
    levene_p: float
    test: str
    stat: float
    p: float
    effect: float | None
    # Power analysis fields
    observed_power: float | None = None
    required_n_per_group_80: float | None = None
    required_n_per_group_90: float | None = None
    effect_size_cohens_d: float | None = None
    effect_size_hedges_g: float | None = None
    mean_diff: float | None = None
    pooled_std: float | None = None

def check_normality(x: pd.Series) -> Tuple[bool, float]:
    x = x.dropna()
    if len(x) < 3:
        return False, np.nan
    try:
        w, p = stats.shapiro(x)
        return p >= 0.05, p
    except Exception:
        return False, np.nan

def hedges_g(x: pd.Series, y: pd.Series) -> float:
    """Calculate Hedges' g effect size for parametric tests."""
    x = x.dropna(); y = y.dropna()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sp = math.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx+ny-2))
    if sp == 0:
        return 0.0
    d = (x.mean() - y.mean()) / sp
    j = 1 - (3/(4*(nx+ny)-9))
    return d * j

def mann_whitney_effect_size(x: pd.Series, y: pd.Series) -> float:
    """Calculate r effect size for Mann-Whitney U test: r = Z/√N."""
    x = x.dropna(); y = y.dropna()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    try:
        u_stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        # Convert U to Z-score
        n = nx + ny
        mean_u = nx * ny / 2
        std_u = math.sqrt(nx * ny * (n + 1) / 12)
        if std_u == 0:
            return 0.0
        z = (u_stat - mean_u) / std_u
        # Calculate r = Z/√N
        r = z / math.sqrt(n)
        return r
    except Exception:
        return np.nan

def calculate_power_analysis(x: pd.Series, y: pd.Series, alpha: float = 0.05) -> Dict[str, float]:
    """
    Calculate power analysis for a two-group comparison.
    
    Parameters:
    -----------
    x, y : pd.Series
        Data for the two groups
    alpha : float
        Significance level (default 0.05)
    
    Returns:
    --------
    dict : Power analysis results including observed power, required sample size for 80% power,
           and effect size
    """
    x = x.dropna()
    y = y.dropna()
    
    if len(x) < 2 or len(y) < 2:
        return {
            'observed_power': np.nan,
            'required_n_per_group_80': np.nan,
            'required_n_per_group_90': np.nan,
            'effect_size_cohens_d': np.nan,
            'effect_size_hedges_g': np.nan,
            'mean_diff': np.nan,
            'pooled_std': np.nan
        }
    
    # Calculate effect sizes
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
    
    # Pooled standard deviation for Cohen's d
    n1, n2 = len(x), len(y)
    pooled_std = np.sqrt(((n1 - 1) * std_x**2 + (n2 - 1) * std_y**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        cohens_d = np.nan
        hedges_g = np.nan
    else:
        cohens_d = (mean_x - mean_y) / pooled_std
        
        # Hedges' g correction for small samples
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohens_d * correction_factor
    
    # Calculate observed power
    try:
        # Use the smaller of the two sample sizes for power calculation
        min_n = min(n1, n2)
        observed_power = ttest_power(effect_size=abs(cohens_d), 
                                   nobs=min_n, 
                                   alpha=alpha, 
                                   alternative='two-sided')
    except:
        observed_power = np.nan
    
    # Calculate required sample size for 80% and 90% power
    try:
        required_n_80 = tt_ind_solve_power(effect_size=abs(cohens_d), 
                                         power=0.8, 
                                         alpha=alpha, 
                                         alternative='two-sided')
        required_n_80 = int(np.ceil(required_n_80))
    except:
        required_n_80 = np.nan
    
    try:
        required_n_90 = tt_ind_solve_power(effect_size=abs(cohens_d), 
                                         power=0.9, 
                                         alpha=alpha, 
                                         alternative='two-sided')
        required_n_90 = int(np.ceil(required_n_90))
    except:
        required_n_90 = np.nan
    
    return {
        'observed_power': observed_power,
        'required_n_per_group_80': required_n_80,
        'required_n_per_group_90': required_n_90,
        'effect_size_cohens_d': cohens_d,
        'effect_size_hedges_g': hedges_g,
        'mean_diff': mean_x - mean_y,
        'pooled_std': pooled_std
    }

def create_power_analysis_summary(stats_files: Dict[str, str], output_file: str = "power_analysis_summary.xlsx"):
    """
    Create a comprehensive Excel summary of power analysis results.
    
    Parameters:
    -----------
    stats_files : dict
        Dictionary mapping analysis type to CSV file path
    output_file : str
        Output Excel file path
    """
    # Create workbook
    wb = openpyxl.Workbook()
    
    # Remove default sheet
    wb.remove(wb.active)
    
    # Define analysis types and their display names
    analysis_types = {
        'normalized': 'Normalized Data',
        'ratio': 'Ratio Data', 
        'proportion': 'Proportion Data',
        'RAW': 'Raw Data'
    }
    
    # Create summary sheet
    summary_ws = wb.create_sheet("Power Analysis Summary", 0)
    
    # Summary headers
    summary_headers = [
        'Analysis Type', 'Region/Layer', 'Current n1', 'Current n2', 
        'Observed Power', 'Effect Size (Cohen\'s d)', 'Effect Size (Hedges\' g)',
        'Required n (80% power)', 'Required n (90% power)', 
        'Mean Difference', 'Pooled Std Dev', 'Significant (p<0.05)',
        'Power Adequate (≥80%)', 'Sample Size Adequate'
    ]
    
    # Write headers
    for col, header in enumerate(summary_headers, 1):
        cell = summary_ws.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")
    
    row = 2
    
    # Process each analysis type
    for analysis_type, display_name in analysis_types.items():
        if analysis_type in stats_files:
            try:
                df = pd.read_csv(stats_files[analysis_type])
                
                for _, row_data in df.iterrows():
                    # Calculate derived metrics
                    power_adequate = "Yes" if not pd.isna(row_data['observed_power']) and row_data['observed_power'] >= 0.8 else "No"
                    sample_adequate = "Yes" if row_data['required_n_per_group_80'] <= min(row_data['n1'], row_data['n2']) else "No"
                    significant = "Yes" if row_data['p'] < 0.05 else "No"
                    
                    # Write data
                    summary_ws.cell(row=row, column=1, value=display_name)
                    summary_ws.cell(row=row, column=2, value=row_data['stratum'])
                    summary_ws.cell(row=row, column=3, value=row_data['n1'])
                    summary_ws.cell(row=row, column=4, value=row_data['n2'])
                    summary_ws.cell(row=row, column=5, value=round(row_data['observed_power'], 3) if not pd.isna(row_data['observed_power']) else "N/A")
                    summary_ws.cell(row=row, column=6, value=round(row_data['effect_size_cohens_d'], 3) if not pd.isna(row_data['effect_size_cohens_d']) else "N/A")
                    summary_ws.cell(row=row, column=7, value=round(row_data['effect_size_hedges_g'], 3) if not pd.isna(row_data['effect_size_hedges_g']) else "N/A")
                    summary_ws.cell(row=row, column=8, value=int(row_data['required_n_per_group_80']) if not pd.isna(row_data['required_n_per_group_80']) else "N/A")
                    summary_ws.cell(row=row, column=9, value=int(row_data['required_n_per_group_90']) if not pd.isna(row_data['required_n_per_group_90']) else "N/A")
                    summary_ws.cell(row=row, column=10, value=round(row_data['mean_diff'], 6) if not pd.isna(row_data['mean_diff']) else "N/A")
                    summary_ws.cell(row=row, column=11, value=round(row_data['pooled_std'], 6) if not pd.isna(row_data['pooled_std']) else "N/A")
                    summary_ws.cell(row=row, column=12, value=significant)
                    summary_ws.cell(row=row, column=13, value=power_adequate)
                    summary_ws.cell(row=row, column=14, value=sample_adequate)
                    
                    # Color code the power adequacy
                    if power_adequate == "Yes":
                        summary_ws.cell(row=row, column=13).fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
                    else:
                        summary_ws.cell(row=row, column=13).fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")
                    
                    # Color code the sample adequacy
                    if sample_adequate == "Yes":
                        summary_ws.cell(row=row, column=14).fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
                    else:
                        summary_ws.cell(row=row, column=14).fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")
                    
                    row += 1
                    
            except Exception as e:
                print(f"Error processing {analysis_type}: {e}")
    
    # Auto-adjust column widths
    for column in summary_ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        summary_ws.column_dimensions[column_letter].width = adjusted_width
    
    # Create individual sheets for each analysis type
    for analysis_type, display_name in analysis_types.items():
        if analysis_type in stats_files:
            try:
                df = pd.read_csv(stats_files[analysis_type])
                ws = wb.create_sheet(display_name)
                
                # Write data to sheet
                for r in dataframe_to_rows(df, index=False, header=True):
                    ws.append(r)
                
                # Format headers
                for cell in ws[1]:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
                    cell.alignment = Alignment(horizontal="center")
                
                # Auto-adjust column widths
                for column in ws.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    ws.column_dimensions[column_letter].width = adjusted_width
                    
            except Exception as e:
                print(f"Error creating sheet for {analysis_type}: {e}")
    
    # Save workbook
    wb.save(output_file)
    print(f"Power analysis summary saved to: {output_file}")
    
    return output_file

def calculate_practical_effect_thresholds(stats_df: pd.DataFrame, max_practical_n: int = 50, 
                                        power: float = 0.8, alpha: float = 0.05) -> pd.DataFrame:
    """
    Calculate practical effect size thresholds - when effects become too small to detect
    with reasonable sample sizes.
    
    Parameters:
    -----------
    stats_df : pd.DataFrame
        Statistics dataframe with effect sizes
    max_practical_n : int
        Maximum practical sample size per group
    power : float
        Desired power level
    alpha : float
        Significance level
    
    Returns:
    --------
    pd.DataFrame with practical thresholds and recommendations
    """
    results = []
    
    for _, row in stats_df.iterrows():
        observed_d = row['effect_size_cohens_d']
        stratum = row['stratum']
        
        if pd.isna(observed_d):
            continue
            
        # Calculate maximum detectable effect size with practical sample size
        try:
            max_detectable_d = tt_ind_solve_power(
                effect_size=None, power=power, alpha=alpha, 
                alternative='two-sided', nobs1=max_practical_n
            )
        except:
            max_detectable_d = np.nan
        
        # Calculate required sample size for observed effect
        try:
            required_n = tt_ind_solve_power(
                effect_size=abs(observed_d), power=power, alpha=alpha,
                alternative='two-sided'
            )
            required_n = int(np.ceil(required_n))
        except:
            required_n = np.nan
        
        # Determine practical categories
        abs_d = abs(observed_d)
        
        if abs_d < 0.2:
            effect_category = "Very Small (impractical)"
            practical = False
            recommendation = "Effect too small to detect practically"
        elif abs_d < 0.5:
            effect_category = "Small (expensive)"
            practical = required_n <= max_practical_n if not pd.isna(required_n) else False
            recommendation = f"Need {required_n} animals per group" if not pd.isna(required_n) else "Cannot calculate"
        elif abs_d < 0.8:
            effect_category = "Medium (moderate cost)"
            practical = required_n <= max_practical_n if not pd.isna(required_n) else False
            recommendation = f"Need {required_n} animals per group" if not pd.isna(required_n) else "Cannot calculate"
        else:
            effect_category = "Large (practical)"
            practical = True
            recommendation = f"Detectable with {required_n} animals per group" if not pd.isna(required_n) else "Detectable"
        
        # Calculate minimum detectable effect (MDE) with current sample size
        current_n = min(row['n1'], row['n2'])
        try:
            mde = tt_ind_solve_power(
                effect_size=None, power=power, alpha=alpha,
                alternative='two-sided', nobs1=current_n
            )
        except:
            mde = np.nan
        
        results.append({
            'stratum': stratum,
            'observed_effect_size': observed_d,
            'abs_effect_size': abs_d,
            'effect_category': effect_category,
            'is_practical': practical,
            'required_n_for_80_power': required_n,
            'max_detectable_d': max_detectable_d,
            'min_detectable_effect_current_n': mde,
            'recommendation': recommendation
        })
    
    return pd.DataFrame(results)

def run_per_stratum_tests(df: pd.DataFrame, conds: List[str], stratum_col: str, value_col: str = "Norm") -> pd.DataFrame:
    results: List[TestResult] = []
    for level, sub in df.groupby(stratum_col):
        x = sub.loc[sub["Condition"] == conds[0], value_col]
        y = sub.loc[sub["Condition"] == conds[1], value_col]
        n1, n2 = x.notna().sum(), y.notna().sum()
        if n1 < 2 or n2 < 2:
            continue
        norm1, _ = check_normality(x)
        norm2, _ = check_normality(y)
        try:
            lev_stat, lev_p = stats.levene(x.dropna(), y.dropna())
        except Exception:
            lev_p = np.nan
        
        # Calculate power analysis
        power_results = calculate_power_analysis(x, y)
        
        if norm1 and norm2:
            equal_var = (lev_p >= 0.05) if not np.isnan(lev_p) else True
            t_stat, p = stats.ttest_ind(x, y, equal_var=equal_var)
            test_name = "Student t" if equal_var else "Welch t"
            eff = hedges_g(x, y)
        else:
            try:
                u_stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                t_stat = u_stat
                test_name = "Mann–Whitney U"
                eff = mann_whitney_effect_size(x, y)
            except Exception:
                t_stat, p, test_name, eff = np.nan, np.nan, "NA", np.nan
        
        results.append(TestResult(
            stratum_col, str(level), n1, n2, norm1, norm2, lev_p,
            test_name, t_stat, p, eff,
            power_results['observed_power'],
            power_results['required_n_per_group_80'],
            power_results['required_n_per_group_90'],
            power_results['effect_size_cohens_d'],
            power_results['effect_size_hedges_g'],
            power_results['mean_diff'],
            power_results['pooled_std']
        ))
    return pd.DataFrame([r.__dict__ for r in results])

def run_per_stratum_tests_forced_mw(df: pd.DataFrame, conds: List[str], stratum_col: str, value_col: str = "Norm") -> pd.DataFrame:
    """Force all comparisons to use Mann-Whitney U test."""
    results: List[TestResult] = []
    for level, sub in df.groupby(stratum_col):
        x = sub.loc[sub["Condition"] == conds[0], value_col]
        y = sub.loc[sub["Condition"] == conds[1], value_col]
        n1, n2 = x.notna().sum(), y.notna().sum()
        if n1 < 2 or n2 < 2:
            continue
        norm1, _ = check_normality(x)
        norm2, _ = check_normality(y)
        try:
            lev_stat, lev_p = stats.levene(x.dropna(), y.dropna())
        except Exception:
            lev_p = np.nan
        
        # Calculate power analysis
        power_results = calculate_power_analysis(x, y)
        
        try:
            u_stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            t_stat = u_stat
            test_name = "Mann–Whitney U (forced)"
            eff = mann_whitney_effect_size(x, y)
        except Exception:
            t_stat, p, test_name, eff = np.nan, np.nan, "NA", np.nan
        
        results.append(TestResult(
            stratum_col, str(level), n1, n2, norm1, norm2, lev_p,
            test_name, t_stat, p, eff,
            power_results['observed_power'],
            power_results['required_n_per_group_80'],
            power_results['required_n_per_group_90'],
            power_results['effect_size_cohens_d'],
            power_results['effect_size_hedges_g'],
            power_results['mean_diff'],
            power_results['pooled_std']
        ))
    return pd.DataFrame([r.__dict__ for r in results])

def run_per_stratum_tests_forced_st(df: pd.DataFrame, conds: List[str], stratum_col: str, value_col: str = "Norm") -> pd.DataFrame:
    """Force all comparisons to use Student's t-test."""
    results: List[TestResult] = []
    for level, sub in df.groupby(stratum_col):
        x = sub.loc[sub["Condition"] == conds[0], value_col]
        y = sub.loc[sub["Condition"] == conds[1], value_col]
        n1, n2 = x.notna().sum(), y.notna().sum()
        if n1 < 2 or n2 < 2:
            continue
        norm1, _ = check_normality(x)
        norm2, _ = check_normality(y)
        try:
            lev_stat, lev_p = stats.levene(x.dropna(), y.dropna())
        except Exception:
            lev_p = np.nan
        equal_var = (lev_p >= 0.05) if not np.isnan(lev_p) else True
        
        # Calculate power analysis
        power_results = calculate_power_analysis(x, y)
        
        try:
            t_stat, p = stats.ttest_ind(x, y, equal_var=equal_var)
            test_name = "Student t (forced)" if equal_var else "Welch t (forced)"
            eff = hedges_g(x, y)
        except Exception:
            t_stat, p, test_name, eff = np.nan, np.nan, "NA", np.nan
        
        results.append(TestResult(
            stratum_col, str(level), n1, n2, norm1, norm2, lev_p,
            test_name, t_stat, p, eff,
            power_results['observed_power'],
            power_results['required_n_per_group_80'],
            power_results['required_n_per_group_90'],
            power_results['effect_size_cohens_d'],
            power_results['effect_size_hedges_g'],
            power_results['mean_diff'],
            power_results['pooled_std']
        ))
    return pd.DataFrame([r.__dict__ for r in results])

# --------------
# Plotting
# --------------

def _box_by_condition(ax, sub: pd.DataFrame, title: str, p_value: float = None, test_name: str = None):
    conds = list(sub["Condition"].astype(str).unique())
    conds.sort()
    data = [sub.loc[sub["Condition"] == c, "Norm"].dropna().values for c in conds]
    
    # Create box plot
    bp = ax.boxplot(data, labels=conds, patch_artist=False, showfliers=False)
    
    # Add individual points with animal labels
    for i, cond in enumerate(conds):
        cond_data = sub.loc[sub["Condition"] == cond, ["Norm", "SampleID"]].dropna()
        if not cond_data.empty:
            # Add jitter to x-coordinates to spread points (fixed seed for reproducibility)
            np.random.seed(42 + i)  # Different seed for each condition
            x_pos = np.random.normal(i+1, 0.1, len(cond_data))
            ax.scatter(x_pos, cond_data["Norm"], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            # Add animal ID labels
            for j, (_, row) in enumerate(cond_data.iterrows()):
                ax.annotate(row["SampleID"], 
                           (x_pos[j], row["Norm"]),
                           xytext=(0, 5), textcoords='offset points',
                           fontsize=8, ha='center', va='bottom')
    
    # Add statistical comparison if p-value provided
    if p_value is not None and len(conds) == 2:
        # Calculate y position for comparison line
        y_max = max([max(d) if len(d) > 0 else 0 for d in data])
        y_min = min([min(d) if len(d) > 0 else 0 for d in data])
        y_range = y_max - y_min
        line_y = y_max + 0.05 * y_range  # Reduced spacing
        
        # Draw comparison line
        ax.plot([1, 2], [line_y, line_y], 'k-', linewidth=1)
        ax.plot([1, 1], [line_y - 0.01*y_range, line_y], 'k-', linewidth=1)
        ax.plot([2, 2], [line_y - 0.01*y_range, line_y], 'k-', linewidth=1)
        
        # Add compact p-value text
        if p_value < 0.001:
            p_text = "p < 0.001"
        elif p_value < 0.01:
            p_text = f"p = {p_value:.3f}"
        elif p_value < 0.05:
            p_text = f"p = {p_value:.3f}"
        else:
            p_text = f"p = {p_value:.2f}"
            
        ax.text(1.5, line_y + 0.01*y_range, p_text, 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
    
    ax.set_ylabel("(cells/starter)/area")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

def _box_by_condition_ratio(ax, sub: pd.DataFrame, title: str, p_value: float = None, test_name: str = None):
    """Box plot helper for ratio data (cells/starter)."""
    conds = list(sub["Condition"].astype(str).unique())
    conds.sort()
    data = [sub.loc[sub["Condition"] == c, "Ratio"].dropna().values for c in conds]
    
    # Create box plot
    bp = ax.boxplot(data, labels=conds, patch_artist=False, showfliers=False)
    
    # Add individual points with animal labels
    for i, cond in enumerate(conds):
        cond_data = sub.loc[sub["Condition"] == cond, ["Ratio", "SampleID"]].dropna()
        if not cond_data.empty:
            # Add jitter to x-coordinates to spread points (fixed seed for reproducibility)
            np.random.seed(42 + i)  # Different seed for each condition
            x_pos = np.random.normal(i+1, 0.1, len(cond_data))
            ax.scatter(x_pos, cond_data["Ratio"], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            # Add animal ID labels
            for j, (_, row) in enumerate(cond_data.iterrows()):
                ax.annotate(row["SampleID"], 
                           (x_pos[j], row["Ratio"]),
                           xytext=(0, 5), textcoords='offset points',
                           fontsize=8, ha='center', va='bottom')
    
    # Add statistical comparison if p-value provided
    if p_value is not None and len(conds) == 2:
        # Calculate y position for comparison line
        y_max = max([max(d) if len(d) > 0 else 0 for d in data])
        y_min = min([min(d) if len(d) > 0 else 0 for d in data])
        y_range = y_max - y_min
        line_y = y_max + 0.05 * y_range  # Reduced spacing
        
        # Draw comparison line
        ax.plot([1, 2], [line_y, line_y], 'k-', linewidth=1)
        ax.plot([1, 1], [line_y - 0.01*y_range, line_y], 'k-', linewidth=1)
        ax.plot([2, 2], [line_y - 0.01*y_range, line_y], 'k-', linewidth=1)
        
        # Add compact p-value text
        if p_value < 0.001:
            p_text = "p < 0.001"
        elif p_value < 0.01:
            p_text = f"p = {p_value:.3f}"
        elif p_value < 0.05:
            p_text = f"p = {p_value:.3f}"
        else:
            p_text = f"p = {p_value:.2f}"
            
        ax.text(1.5, line_y + 0.01*y_range, p_text, 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
    
    ax.set_ylabel("cells/starter")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

def plot_long_distance(df: pd.DataFrame, stats_df: pd.DataFrame, savepath: str | None = None):
    areas = [a for a in LONG_DISTANCE_AREAS if a in set(df["Region"].astype(str))]
    if not areas:
        return
    
    # Calculate global Y-axis range for consistent scaling
    all_values = df["Norm"].dropna()
    y_min = all_values.min()
    y_max = all_values.max()
    y_range = y_max - y_min
    y_padding = 0.1 * y_range
    y_lim = (y_min - y_padding, y_max + y_padding)
    
    fig, axes = plt.subplots(1, len(areas), figsize=(FIGSIZE[0]*len(areas), FIGSIZE[1]), dpi=PLOT_DPI)
    if len(areas) == 1:
        axes = [axes]
    
    for ax, area in zip(axes, areas):
        # Get p-value for this area
        area_stats = stats_df[stats_df["stratum"] == area]
        p_val = area_stats["p"].iloc[0] if not area_stats.empty else None
        test_name = area_stats["test"].iloc[0] if not area_stats.empty else None
        
        _box_by_condition(ax, df[df["Region"].astype(str) == area], f"{area}", p_val, test_name)
        ax.set_ylim(y_lim)  # Set consistent Y-axis range
    
    fig.suptitle("Long-distance regions: normalized counts", y=1.05)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

def plot_layers(df: pd.DataFrame, stats_df: pd.DataFrame, savepath: str | None = None):
    layers = [l for l in sorted(df["Layer"].dropna().astype(str).unique())]
    if not layers:
        return
    
    # Calculate global Y-axis range for consistent scaling
    all_values = df["Norm"].dropna()
    y_min = all_values.min()
    y_max = all_values.max()
    y_range = y_max - y_min
    y_padding = 0.1 * y_range
    y_lim = (y_min - y_padding, y_max + y_padding)
    
    fig, axes = plt.subplots(1, len(layers), figsize=(FIGSIZE[0]*len(layers), FIGSIZE[1]), dpi=PLOT_DPI)
    if len(layers) == 1:
        axes = [axes]
    
    for ax, layer in zip(axes, layers):
        # Get p-value for this layer
        layer_stats = stats_df[stats_df["stratum"] == layer]
        p_val = layer_stats["p"].iloc[0] if not layer_stats.empty else None
        test_name = layer_stats["test"].iloc[0] if not layer_stats.empty else None
        
        _box_by_condition(ax, df[df["Layer"].astype(str) == layer], f"Layer {layer}", p_val, test_name)
        ax.set_ylim(y_lim)  # Set consistent Y-axis range
    
    fig.suptitle("Local layers at injection site: normalized counts", y=1.05)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

def plot_significance_effect_size(stats_df: pd.DataFrame, savepath: str | None = None):
    """Create a 4-quadrant plot showing significance vs effect size."""
    if stats_df.empty:
        return
    
    # Prepare data
    df_plot = stats_df.copy()
    df_plot['-log10_p'] = -np.log10(df_plot['p'].replace(0, 1e-10))  # Handle p=0
    df_plot['effect_size'] = df_plot['effect'].fillna(0)  # Use actual effect size, not absolute
    
    # Create the plot with extra space for legend and labels
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=PLOT_DPI)
    
    # Define significance threshold (p < 0.05)
    sig_threshold = -np.log10(0.05)
    
    # Define effect size categories (Cohen's conventions)
    small_effect = 0.2
    medium_effect = 0.5
    large_effect = 0.8
    
    # Color points based on significance and effect size
    colors = []
    sizes = []
    alphas = []
    
    for _, row in df_plot.iterrows():
        p_val = row['p']
        effect = row['effect_size']
        abs_eff = abs(effect)
        
        # Determine color based on significance and effect size
        if p_val < 0.05:
            if abs_eff >= large_effect:
                colors.append('red')  # Significant + Large effect
            elif abs_eff >= medium_effect:
                colors.append('orange')  # Significant + Medium effect
            else:
                colors.append('yellow')  # Significant + Small effect
        else:
            colors.append('lightgray')  # Not significant
        
        # Size based on effect size
        if abs_eff >= large_effect:
            sizes.append(150)
        elif abs_eff >= medium_effect:
            sizes.append(100)
        else:
            sizes.append(50)
        
        # Alpha based on significance
        alphas.append(0.8 if p_val < 0.05 else 0.4)
    
    # Create scatter plot
    scatter = ax.scatter(df_plot['effect_size'], df_plot['-log10_p'], 
                        c=colors, s=sizes, alpha=alphas, edgecolors='black', linewidth=0.5)
    
    # Add quadrant lines
    ax.axhline(y=sig_threshold, color='black', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=small_effect, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.axvline(x=medium_effect, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.axvline(x=large_effect, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    
    # Add labels for each point
    for i, row in df_plot.iterrows():
        ax.annotate(f"{row['stratum']}", 
                   (row['effect_size'], row['-log10_p']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, ha='left')
    
    # Set axis labels and limits
    ax.set_xlabel('Effect Size (Hedges\' g)', fontsize=12)
    ax.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax.set_title('Significance vs Effect Size\n(4-Quadrant Analysis)', fontsize=14, fontweight='bold')
    
    # Set consistent axis limits across all plots
    ax.set_xlim(-5, 5)  # Effect size from -5 to 5 (covers all data points with padding)
    ax.set_ylim(0, 3)   # Y-axis from 0 to 3 (significance at 1.3 is near middle)
    
    # Add quadrant labels outside the plot area
    ax.text(1.02, 0.95, 'Significant\nLarge Effect', transform=ax.transAxes, 
            ha='left', va='top', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
    ax.text(-0.02, 0.95, 'Significant\nSmall Effect', transform=ax.transAxes, 
            ha='right', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    ax.text(1.02, 0.05, 'Not Significant\nLarge Effect', transform=ax.transAxes, 
            ha='left', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))
    ax.text(-0.02, 0.05, 'Not Significant\nSmall Effect', transform=ax.transAxes, 
            ha='right', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))
    
    # Add effect size reference lines
    y_top = ax.get_ylim()[1]
    ax.text(small_effect, y_top*0.9, 'Small\n(0.2)', ha='center', va='top', 
            fontsize=8, color='gray')
    ax.text(medium_effect, y_top*0.9, 'Medium\n(0.5)', ha='center', va='top', 
            fontsize=8, color='gray')
    ax.text(large_effect, y_top*0.9, 'Large\n(0.8)', ha='center', va='top', 
            fontsize=8, color='gray')
    
    # Add note if any points are outside the visible range
    x_min, x_max = df_plot['effect_size'].min(), df_plot['effect_size'].max()
    if x_min < -5 or x_max > 5:
        clipped_count = len(df_plot[(df_plot['effect_size'] < -5) | (df_plot['effect_size'] > 5)])
        ax.text(0.5, 0.02, f'Note: {clipped_count} points outside x-axis range', 
                transform=ax.transAxes, ha='center', va='bottom', 
                fontsize=8, style='italic', color='red')
    
    # Fixed scales - no log scaling needed
    
    # Add legend outside the plot area
    legend_elements = [
        plt.scatter([], [], c='red', s=150, label='Significant + Large Effect', edgecolors='black'),
        plt.scatter([], [], c='orange', s=100, label='Significant + Medium Effect', edgecolors='black'),
        plt.scatter([], [], c='yellow', s=50, label='Significant + Small Effect', edgecolors='black'),
        plt.scatter([], [], c='lightgray', s=50, label='Not Significant', edgecolors='black', alpha=0.4)
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout to prevent clipping of legend and labels
    plt.tight_layout()
    plt.subplots_adjust(right=0.75, left=0.15)  # Make room for legend and quadrant labels
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=PLOT_DPI)
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

# -----------------
# Ratio plotting functions (cells/starter only)
# -----------------

def plot_long_distance_ratio(df: pd.DataFrame, stats_df: pd.DataFrame, savepath: str | None = None):
    """Plot long-distance regions for ratio data (cells/starter)."""
    areas = [a for a in LONG_DISTANCE_AREAS if a in set(df["Region"].astype(str))]
    if not areas:
        return
    
    # Calculate global Y-axis range for consistent scaling
    all_values = df["Ratio"].dropna()
    y_min = all_values.min()
    y_max = all_values.max()
    y_range = y_max - y_min
    y_padding = 0.1 * y_range
    y_lim = (y_min - y_padding, y_max + y_padding)
    
    fig, axes = plt.subplots(1, len(areas), figsize=(FIGSIZE[0]*len(areas), FIGSIZE[1]), dpi=PLOT_DPI)
    if len(areas) == 1:
        axes = [axes]
    
    for ax, area in zip(axes, areas):
        # Get p-value for this area
        area_stats = stats_df[stats_df["stratum"] == area]
        p_val = area_stats["p"].iloc[0] if not area_stats.empty else None
        test_name = area_stats["test"].iloc[0] if not area_stats.empty else None
        
        _box_by_condition_ratio(ax, df[df["Region"].astype(str) == area], f"{area}", p_val, test_name)
        ax.set_ylim(y_lim)  # Set consistent Y-axis range
    
    fig.suptitle("Long-distance regions: cells/starter ratio", y=1.05)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

def plot_layers_ratio(df: pd.DataFrame, stats_df: pd.DataFrame, savepath: str | None = None):
    """Plot layers for ratio data (cells/starter)."""
    layers = [l for l in sorted(df["Layer"].dropna().astype(str).unique())]
    if not layers:
        return
    
    # Calculate global Y-axis range for consistent scaling
    all_values = df["Ratio"].dropna()
    y_min = all_values.min()
    y_max = all_values.max()
    y_range = y_max - y_min
    y_padding = 0.1 * y_range
    y_lim = (y_min - y_padding, y_max + y_padding)
    
    fig, axes = plt.subplots(1, len(layers), figsize=(FIGSIZE[0]*len(layers), FIGSIZE[1]), dpi=PLOT_DPI)
    if len(layers) == 1:
        axes = [axes]
    
    for ax, layer in zip(axes, layers):
        # Get p-value for this layer
        layer_stats = stats_df[stats_df["stratum"] == layer]
        p_val = layer_stats["p"].iloc[0] if not layer_stats.empty else None
        test_name = layer_stats["test"].iloc[0] if not layer_stats.empty else None
        
        _box_by_condition_ratio(ax, df[df["Layer"].astype(str) == layer], f"Layer {layer}", p_val, test_name)
        ax.set_ylim(y_lim)  # Set consistent Y-axis range
    
    fig.suptitle("Local layers at injection site: cells/starter ratio", y=1.05)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

# -----------------
# Proportion plotting functions
# -----------------

def _box_by_condition_proportion(ax, sub: pd.DataFrame, title: str, p_value: float = None, test_name: str = None):
    """Box plot helper for proportion data (percentages)."""
    conds = list(sub["Condition"].astype(str).unique())
    conds.sort()
    data = [sub.loc[sub["Condition"] == c, "Proportion"].dropna().values for c in conds]
    
    # Create box plot
    bp = ax.boxplot(data, labels=conds, patch_artist=False, showfliers=False)
    
    # Add individual points with animal labels
    for i, cond in enumerate(conds):
        cond_data = sub.loc[sub["Condition"] == cond, ["Proportion", "SampleID"]].dropna()
        if not cond_data.empty:
            # Add jitter to x-coordinates to spread points (fixed seed for reproducibility)
            np.random.seed(42 + i)  # Different seed for each condition
            x_pos = np.random.normal(i+1, 0.1, len(cond_data))
            ax.scatter(x_pos, cond_data["Proportion"], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            # Add animal ID labels
            for j, (_, row) in enumerate(cond_data.iterrows()):
                ax.annotate(row["SampleID"], 
                           (x_pos[j], row["Proportion"]),
                           xytext=(0, 5), textcoords='offset points',
                           fontsize=8, ha='center', va='bottom')
    
    # Add statistical comparison if p-value provided
    if p_value is not None and len(conds) == 2:
        # Calculate y position for comparison line
        y_max = max([max(d) if len(d) > 0 else 0 for d in data])
        y_min = min([min(d) if len(d) > 0 else 0 for d in data])
        y_range = y_max - y_min
        line_y = y_max + 0.05 * y_range  # Reduced spacing
        
        # Draw comparison line
        ax.plot([1, 2], [line_y, line_y], 'k-', linewidth=1)
        ax.plot([1, 1], [line_y - 0.01*y_range, line_y], 'k-', linewidth=1)
        ax.plot([2, 2], [line_y - 0.01*y_range, line_y], 'k-', linewidth=1)
        
        # Add compact p-value text
        if p_value < 0.001:
            p_text = "p < 0.001"
        elif p_value < 0.01:
            p_text = f"p = {p_value:.3f}"
        elif p_value < 0.05:
            p_text = f"p = {p_value:.3f}"
        else:
            p_text = f"p = {p_value:.2f}"
            
        ax.text(1.5, line_y + 0.01*y_range, p_text, 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
    
    ax.set_ylabel("Proportion (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

def plot_long_distance_proportion(df: pd.DataFrame, stats_df: pd.DataFrame, savepath: str | None = None):
    """Plot long-distance regions for proportion data (percentages)."""
    areas = [a for a in LONG_DISTANCE_AREAS if a in set(df["Region"].astype(str))]
    if not areas:
        return
    
    # Calculate global Y-axis range for consistent scaling (0-100%)
    y_lim = (0, 100)
    
    fig, axes = plt.subplots(1, len(areas), figsize=(FIGSIZE[0]*len(areas), FIGSIZE[1]), dpi=PLOT_DPI)
    if len(areas) == 1:
        axes = [axes]
    
    for ax, area in zip(axes, areas):
        # Get p-value for this area
        area_stats = stats_df[stats_df["stratum"] == area]
        p_val = area_stats["p"].iloc[0] if not area_stats.empty else None
        test_name = area_stats["test"].iloc[0] if not area_stats.empty else None
        
        _box_by_condition_proportion(ax, df[df["Region"].astype(str) == area], f"{area}", p_val, test_name)
        ax.set_ylim(y_lim)  # Set consistent Y-axis range (0-100%)
    
    fig.suptitle("Long-distance regions: proportion of total (%)", y=1.05)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

def plot_layers_proportion(df: pd.DataFrame, stats_df: pd.DataFrame, savepath: str | None = None):
    """Plot layers for proportion data (percentages)."""
    layers = [l for l in sorted(df["Layer"].dropna().astype(str).unique())]
    if not layers:
        return
    
    # Calculate global Y-axis range for consistent scaling (0-100%)
    y_lim = (0, 100)
    
    fig, axes = plt.subplots(1, len(layers), figsize=(FIGSIZE[0]*len(layers), FIGSIZE[1]), dpi=PLOT_DPI)
    if len(layers) == 1:
        axes = [axes]
    
    for ax, layer in zip(axes, layers):
        # Get p-value for this layer
        layer_stats = stats_df[stats_df["stratum"] == layer]
        p_val = layer_stats["p"].iloc[0] if not layer_stats.empty else None
        test_name = layer_stats["test"].iloc[0] if not layer_stats.empty else None
        
        _box_by_condition_proportion(ax, df[df["Layer"].astype(str) == layer], f"Layer {layer}", p_val, test_name)
        ax.set_ylim(y_lim)  # Set consistent Y-axis range (0-100%)
    
    fig.suptitle("Local layers at injection site: proportion of total (%)", y=1.05)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

# -----------------
# Raw counts plotting functions (no normalization)
# -----------------

def _box_by_condition_raw(ax, sub: pd.DataFrame, title: str, p_value: float = None, test_name: str = None):
    """Box plot helper for raw count data."""
    conds = list(sub["Condition"].astype(str).unique())
    conds.sort()
    data = [sub.loc[sub["Condition"] == c, "Cells"].dropna().values for c in conds]
    
    # Create box plot
    bp = ax.boxplot(data, labels=conds, patch_artist=False, showfliers=False)
    
    # Add individual points with animal labels
    for i, cond in enumerate(conds):
        cond_data = sub.loc[sub["Condition"] == cond, ["Cells", "SampleID"]].dropna()
        if not cond_data.empty:
            # Add jitter to x-coordinates to spread points (fixed seed for reproducibility)
            np.random.seed(42 + i)  # Different seed for each condition
            x_pos = np.random.normal(i+1, 0.1, len(cond_data))
            ax.scatter(x_pos, cond_data["Cells"], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            # Add animal ID labels
            for j, (_, row) in enumerate(cond_data.iterrows()):
                ax.annotate(row["SampleID"], 
                           (x_pos[j], row["Cells"]),
                           xytext=(0, 5), textcoords='offset points',
                           fontsize=8, ha='center', va='bottom')
    
    # Add statistical comparison if p-value provided
    if p_value is not None and len(conds) == 2:
        # Calculate y position for comparison line
        y_max = max([max(d) if len(d) > 0 else 0 for d in data])
        y_min = min([min(d) if len(d) > 0 else 0 for d in data])
        y_range = y_max - y_min
        line_y = y_max + 0.05 * y_range  # Reduced spacing
        
        # Draw comparison line
        ax.plot([1, 2], [line_y, line_y], 'k-', linewidth=1)
        ax.plot([1, 1], [line_y - 0.01*y_range, line_y], 'k-', linewidth=1)
        ax.plot([2, 2], [line_y - 0.01*y_range, line_y], 'k-', linewidth=1)
        
        # Add compact p-value text
        if p_value < 0.001:
            p_text = "p < 0.001"
        elif p_value < 0.01:
            p_text = f"p = {p_value:.3f}"
        elif p_value < 0.05:
            p_text = f"p = {p_value:.3f}"
        else:
            p_text = f"p = {p_value:.2f}"
            
        ax.text(1.5, line_y + 0.01*y_range, p_text, 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
    
    ax.set_ylabel("Raw Cell Counts")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

def plot_long_distance_raw(df: pd.DataFrame, stats_df: pd.DataFrame, savepath: str | None = None):
    """Plot long-distance regions for raw count data."""
    areas = [a for a in LONG_DISTANCE_AREAS if a in set(df["Region"].astype(str))]
    if not areas:
        return
    
    # Calculate global Y-axis range for consistent scaling
    all_values = df["Cells"].dropna()
    y_min = all_values.min()
    y_max = all_values.max()
    y_range = y_max - y_min
    y_padding = 0.1 * y_range
    y_lim = (y_min - y_padding, y_max + y_padding)
    
    fig, axes = plt.subplots(1, len(areas), figsize=(FIGSIZE[0]*len(areas), FIGSIZE[1]), dpi=PLOT_DPI)
    if len(areas) == 1:
        axes = [axes]
    
    for ax, area in zip(axes, areas):
        # Get p-value for this area
        area_stats = stats_df[stats_df["stratum"] == area]
        p_val = area_stats["p"].iloc[0] if not area_stats.empty else None
        test_name = area_stats["test"].iloc[0] if not area_stats.empty else None
        
        _box_by_condition_raw(ax, df[df["Region"].astype(str) == area], f"{area}", p_val, test_name)
        ax.set_ylim(y_lim)  # Set consistent Y-axis range
    
    fig.suptitle("Long-distance regions: raw cell counts", y=1.05)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

def plot_layers_raw(df: pd.DataFrame, stats_df: pd.DataFrame, savepath: str | None = None):
    """Plot layers for raw count data."""
    layers = [l for l in sorted(df["Layer"].dropna().astype(str).unique())]
    if not layers:
        return
    
    # Calculate global Y-axis range for consistent scaling
    all_values = df["Cells"].dropna()
    y_min = all_values.min()
    y_max = all_values.max()
    y_range = y_max - y_min
    y_padding = 0.1 * y_range
    y_lim = (y_min - y_padding, y_max + y_padding)
    
    fig, axes = plt.subplots(1, len(layers), figsize=(FIGSIZE[0]*len(layers), FIGSIZE[1]), dpi=PLOT_DPI)
    if len(layers) == 1:
        axes = [axes]
    
    for ax, layer in zip(axes, layers):
        # Get p-value for this layer
        layer_stats = stats_df[stats_df["stratum"] == layer]
        p_val = layer_stats["p"].iloc[0] if not layer_stats.empty else None
        test_name = layer_stats["test"].iloc[0] if not layer_stats.empty else None
        
        _box_by_condition_raw(ax, df[df["Layer"].astype(str) == layer], f"Layer {layer}", p_val, test_name)
        ax.set_ylim(y_lim)  # Set consistent Y-axis range
    
    fig.suptitle("Local layers at injection site: raw cell counts", y=1.05)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

# -----------------
# John's proportion plotting functions
# -----------------

def _box_by_condition_john_proportion(ax, sub: pd.DataFrame, title: str, p_value: float = None, test_name: str = None):
    """Box plot helper for John's proportion data (percentages)."""
    conds = list(sub["Condition"].astype(str).unique())
    conds.sort()
    data = [sub.loc[sub["Condition"] == c, "John_Proportion"].dropna().values for c in conds]
    
    # Create box plot
    bp = ax.boxplot(data, labels=conds, patch_artist=False, showfliers=False)
    
    # Add individual points with animal labels
    for i, cond in enumerate(conds):
        cond_data = sub.loc[sub["Condition"] == cond, ["John_Proportion", "SampleID"]].dropna()
        if not cond_data.empty:
            # Add jitter to x-coordinates to spread points (fixed seed for reproducibility)
            np.random.seed(42 + i)  # Different seed for each condition
            x_pos = np.random.normal(i+1, 0.1, len(cond_data))
            ax.scatter(x_pos, cond_data["John_Proportion"], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            # Add animal ID labels
            for j, (_, row) in enumerate(cond_data.iterrows()):
                ax.annotate(row["SampleID"], 
                           (x_pos[j], row["John_Proportion"]),
                           xytext=(0, 5), textcoords='offset points',
                           fontsize=8, ha='center', va='bottom')
    
    # Add statistical comparison if p-value provided
    if p_value is not None and len(conds) == 2:
        # Calculate y position for comparison line
        y_max = max([max(d) if len(d) > 0 else 0 for d in data])
        y_min = min([min(d) if len(d) > 0 else 0 for d in data])
        y_range = y_max - y_min
        line_y = y_max + 0.05 * y_range  # Reduced spacing
        
        # Draw comparison line
        ax.plot([1, 2], [line_y, line_y], 'k-', linewidth=1)
        ax.plot([1, 1], [line_y - 0.01*y_range, line_y], 'k-', linewidth=1)
        ax.plot([2, 2], [line_y - 0.01*y_range, line_y], 'k-', linewidth=1)
        
        # Add compact p-value text
        if p_value < 0.001:
            p_text = "p < 0.001"
        elif p_value < 0.01:
            p_text = f"p = {p_value:.3f}"
        elif p_value < 0.05:
            p_text = f"p = {p_value:.3f}"
        else:
            p_text = f"p = {p_value:.2f}"
            
        ax.text(1.5, line_y + 0.01*y_range, p_text, 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
    
    ax.set_ylabel("John's Proportion (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

def plot_john_proportions_bar_chart(df: pd.DataFrame, proportion_type='all_non_starter', savepath: str | None = None):
    """
    Create a bar chart showing John's proportion calculations with individual animals plotted,
    grouped by injection scheme. This mimics the Rapid Rabies Methods paper approach.
    """
    # Define area columns (matching John's approach)
    v1_areas = ['V1 L2/3', 'V1 L4', 'V1 L5', 'V1 L6a', 'V1 L6b']
    non_v1_areas = ['V2M', 'V2L', 'dLGN']
    all_areas = v1_areas + non_v1_areas
    
    # Create area counts with combined L2/3 (matching John's logic)
    area_counts = df[['V1 L4', 'V1 L5', 'V1 L6a', 'V1 L6b', 'V2M', 'V2L', 'dLGN']].copy()
    
    # Handle different column formats
    if 'V1 L2/3 Upper' in df.columns and 'V1 L2/3 Lower' in df.columns:
        # Original format with separate Upper/Lower columns
        area_counts['V1 L2/3'] = df['V1 L2/3 Upper'] + df['V1 L2/3 Lower']
    elif 'V1 L2/3' in df.columns:
        # Already combined format
        area_counts['V1 L2/3'] = df['V1 L2/3']
    else:
        raise ValueError("Could not find V1 L2/3 columns in expected formats")
    
    # Calculate proportions for each injection scheme group
    injection_schemes = df['Injection Scheme'].unique()
    group_proportions = {}
    
    for scheme in injection_schemes:
        scheme_mask = df['Injection Scheme'] == scheme
        scheme_area_counts = area_counts[scheme_mask]
        
        if proportion_type == 'all_non_starter':
            # Calculate proportions relative to all non-starter inputs
            total_non_starter = scheme_area_counts[all_areas].sum(axis=1)
            proportions = scheme_area_counts[all_areas].div(total_non_starter, axis=0)
            
        elif proportion_type == 'local_vs_long_distance':
            # Calculate local (V1) and long-distance (non-V1) totals
            local_total = scheme_area_counts[v1_areas].sum(axis=1)
            long_distance_total = scheme_area_counts[non_v1_areas].sum(axis=1)
            
            # Initialize proportions dataframe
            proportions = pd.DataFrame(index=scheme_area_counts.index, columns=all_areas)
            
            # Calculate proportions for V1 areas (relative to local inputs)
            for area in v1_areas:
                proportions[area] = scheme_area_counts[area] / local_total
            
            # Calculate proportions for non-V1 areas (relative to long-distance inputs)
            for area in non_v1_areas:
                proportions[area] = scheme_area_counts[area] / long_distance_total
        
        # Set Animal ID as index and handle NaN values
        proportions.index = df[scheme_mask]['Animal ID']
        proportions = proportions.fillna(0)
        group_proportions[scheme] = proportions
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 8), dpi=PLOT_DPI)
    
    # Define colors for injection schemes
    scheme_colors = {
        'helper + rabies coinjected (rapid tracing)': 'lightblue',
        'helper + rabies seperated': 'lightcoral',
        'coinjected': 'lightblue',
        'separated': 'lightcoral'
    }
    
    # Calculate bar positions
    n_areas = len(all_areas)
    n_schemes = len(injection_schemes)
    bar_width = 0.35
    x_pos = np.arange(n_areas)
    
    # Create bars for each injection scheme
    for i, scheme in enumerate(injection_schemes):
        proportions = group_proportions[scheme]
        means = proportions.mean()
        stds = proportions.std()
        
        # Position bars side by side
        x_offset = (i - (n_schemes-1)/2) * bar_width
        
        # Create bars
        bars = ax.bar(x_pos + x_offset, means, bar_width, yerr=stds, capsize=5, alpha=0.7,
                     color=scheme_colors.get(scheme, 'gray'), 
                     edgecolor='black', linewidth=1,
                     label=scheme)
        
        # Add individual animal data points
        for j, area in enumerate(all_areas):
            for animal_id in proportions.index:
                # Add some jitter to x position for visibility (fixed seed for reproducibility)
                np.random.seed(42 + j)  # Different seed for each area
                jitter = np.random.normal(0, 0.05)
                ax.scatter(j + x_offset + jitter, proportions.loc[animal_id, area], 
                          color='black', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Brain Area', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion of Inputs', fontsize=12, fontweight='bold')
    
    # Set title based on proportion type
    if proportion_type == 'all_non_starter':
        title = 'John\'s Method: Proportion of All Non-Starter Inputs by Brain Area\n(Comparison by Injection Scheme)'
    elif proportion_type == 'local_vs_long_distance':
        title = 'John\'s Method: Proportion of Local vs Long-Distance Inputs by Brain Area\n(V1 areas: % of local inputs, V2M/V2L/dLGN: % of long-distance inputs)\n(Comparison by Injection Scheme)'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_areas, rotation=45, ha='right')
    
    # Calculate statistics comparing injection schemes (John's simple t-test approach)
    p_values = {}
    for area in all_areas:
        if len(injection_schemes) == 2:
            scheme1, scheme2 = injection_schemes
            group1_data = group_proportions[scheme1][area]
            group2_data = group_proportions[scheme2][area]
            
            # Perform independent t-test (John's approach)
            t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
            p_values[area] = p_val
    
    # Add statistical significance annotations
    all_means = pd.concat([group_proportions[scheme].mean() for scheme in injection_schemes])
    all_stds = pd.concat([group_proportions[scheme].std() for scheme in injection_schemes])
    max_height = max(all_means + all_stds)
    
    for i, area in enumerate(all_areas):
        if area in p_values:
            p_val = p_values[area]
            
            # Determine significance level
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'NS'
            
            # Add significance annotation above the bars
            y_pos = max_height * 1.05
            ax.text(i, y_pos, sig_text, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Print p-values to console (John's approach)
    print(f"John's Method - Statistical comparison between injection schemes ({proportion_type}):")
    print("=" * 70)
    for area, p_val in p_values.items():
        if p_val < 0.001:
            sig_level = "p < 0.001 (***)"
        elif p_val < 0.01:
            sig_level = f"p = {p_val:.3f} (**)"
        elif p_val < 0.05:
            sig_level = f"p = {p_val:.3f} (*)"
        else:
            sig_level = f"p = {p_val:.3f} (NS)"
        print(f"{area}: {sig_level}")
    print("\n* p < 0.05, ** p < 0.01, *** p < 0.001, NS = not significant")
    
    ax.set_ylim(0, max(all_means + all_stds) * 1.15)  # Extra space for significance annotations
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Add legend
    ax.legend(title='Injection Scheme', loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=PLOT_DPI)
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

def plot_john_regional_inputs_bar_chart(df: pd.DataFrame, normalize_by_area=False, savepath: str | None = None):
    """
    Create a bar chart showing John's regional inputs (V1, V2M, V2L, dLGN) with individual animals plotted,
    grouped by injection scheme. This mimics the Rapid Rabies Methods paper approach.
    """
    # Define brain regions (local and long-distance)
    brain_regions = ['V1', 'V2M', 'V2L', 'dLGN']
    area_columns = {
        'V1': 'V1 Area (pixels)',
        'V2M': 'V2M Area (pixels)',
        'V2L': 'V2L Area (pixels)', 
        'dLGN': 'dLGN Area (pixels)'
    }
    
    # Calculate counts for all brain regions
    area_counts = pd.DataFrame(index=df.index)
    
    # Calculate total V1 (all layers combined, minus starters)
    if 'V1 L2/3 Upper' in df.columns and 'V1 L2/3 Lower' in df.columns:
        # Original format with separate Upper/Lower columns
        v1_total = df['V1 L2/3 Upper'] + df['V1 L2/3 Lower'] + df['V1 L4'] + df['V1 L5'] + df['V1 L6a'] + df['V1 L6b']
    elif 'V1 L2/3' in df.columns:
        # Already combined format
        v1_total = df['V1 L2/3'] + df['V1 L4'] + df['V1 L5'] + df['V1 L6a'] + df['V1 L6b']
    else:
        raise ValueError("Could not find V1 L2/3 columns in expected formats")
    
    area_counts['V1'] = v1_total - df['V1 Starters']
    
    # Add other regions
    area_counts['V2M'] = df['V2M']
    area_counts['V2L'] = df['V2L'] 
    area_counts['dLGN'] = df['dLGN']
    
    # Normalize by area if requested
    if normalize_by_area:
        for region in brain_regions:
            area_col = area_columns[region]
            area_counts[region] = area_counts[region] / df[area_col]
    
    # Calculate proportions for each injection scheme group
    injection_schemes = df['Injection Scheme'].unique()
    group_proportions = {}
    
    for scheme in injection_schemes:
        scheme_mask = df['Injection Scheme'] == scheme
        scheme_area_counts = area_counts[scheme_mask]
        
        # Calculate proportions across all regions
        total_inputs = scheme_area_counts.sum(axis=1)
        proportions = scheme_area_counts.div(total_inputs, axis=0).fillna(0)
        
        # Set Animal ID as index
        proportions.index = df[scheme_mask]['Animal ID']
        group_proportions[scheme] = proportions
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=PLOT_DPI)
    
    # Define colors for injection schemes
    scheme_colors = {
        'helper + rabies coinjected (rapid tracing)': 'lightblue',
        'helper + rabies seperated': 'lightcoral',
        'coinjected': 'lightblue',
        'separated': 'lightcoral'
    }
    
    # Calculate bar positions
    n_regions = len(brain_regions)
    n_schemes = len(injection_schemes)
    bar_width = 0.35
    x_pos = np.arange(n_regions)
    
    # Create bars for each injection scheme
    for i, scheme in enumerate(injection_schemes):
        proportions = group_proportions[scheme]
        means = proportions.mean()
        stds = proportions.std()
        
        # Position bars side by side
        x_offset = (i - (n_schemes-1)/2) * bar_width
        
        # Create bars
        bars = ax.bar(x_pos + x_offset, means, bar_width, yerr=stds, capsize=5, alpha=0.7,
                     color=scheme_colors.get(scheme, 'gray'), 
                     edgecolor='black', linewidth=1,
                     label=scheme)
        
        # Add individual animal data points
        for j, region in enumerate(brain_regions):
            for animal_id in proportions.index:
                # Add some jitter to x position for visibility (fixed seed for reproducibility)
                np.random.seed(42 + j)  # Different seed for each region
                jitter = np.random.normal(0, 0.05)
                ax.scatter(j + x_offset + jitter, proportions.loc[animal_id, region], 
                          color='black', s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Brain Area', fontsize=12, fontweight='bold')
    
    if normalize_by_area:
        ax.set_ylabel('John\'s Method: Proportion of Inputs\n(Area-Normalized)', fontsize=12, fontweight='bold')
        title = 'John\'s Method: Input Proportions by Brain Area\n(Area-Normalized)\n(Comparison by Injection Scheme)'
    else:
        ax.set_ylabel('John\'s Method: Proportion of Inputs\n(Non-Area-Normalized)', fontsize=12, fontweight='bold')
        title = 'John\'s Method: Input Proportions by Brain Area\n(Non-Area-Normalized)\n(Comparison by Injection Scheme)'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(brain_regions, rotation=0)
    
    # Calculate statistics comparing injection schemes (John's simple t-test approach)
    p_values = {}
    for region in brain_regions:
        if len(injection_schemes) == 2:
            scheme1, scheme2 = injection_schemes
            group1_data = group_proportions[scheme1][region]
            group2_data = group_proportions[scheme2][region]
            
            # Perform independent t-test (John's approach)
            t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
            p_values[region] = p_val
    
    # Add statistical significance annotations
    all_means = pd.concat([group_proportions[scheme].mean() for scheme in injection_schemes])
    all_stds = pd.concat([group_proportions[scheme].std() for scheme in injection_schemes])
    max_height = max(all_means + all_stds)
    
    for i, region in enumerate(brain_regions):
        if region in p_values:
            p_val = p_values[region]
            
            # Determine significance level
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'NS'
            
            # Add significance annotation above the bars
            y_pos = max_height * 1.05
            ax.text(i, y_pos, sig_text, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Print p-values to console (John's approach)
    print(f"John's Method - Statistical comparison between injection schemes (Regional Inputs, area_normalized={normalize_by_area}):")
    print("=" * 80)
    for region, p_val in p_values.items():
        if p_val < 0.001:
            sig_level = "p < 0.001 (***)"
        elif p_val < 0.01:
            sig_level = f"p = {p_val:.3f} (**)"
        elif p_val < 0.05:
            sig_level = f"p = {p_val:.3f} (*)"
        else:
            sig_level = f"p = {p_val:.3f} (NS)"
        print(f"{region}: {sig_level}")
    print("\n* p < 0.05, ** p < 0.01, *** p < 0.001, NS = not significant")
    
    ax.set_ylim(0, max(all_means + all_stds) * 1.15)  # Extra space for significance annotations
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Add legend
    ax.legend(title='Injection Scheme', loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=PLOT_DPI)
        # Also save as SVG
        svg_path = savepath.replace('.png', '.svg')
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

# -----------------
# Main analysis API
# -----------------

def analyze(csv_path: str = "data/rabies_comparison.csv",
            out_prefix: str = "rabies_analysis") -> Dict[str, str]:
    # Read and transform the data
    df_orig = pd.read_csv(csv_path)
    df_long = transform_wide_to_long(df_orig)
    conds = assert_two_conditions(df_long)
    
    # Compute normalized, ratio, and proportion data
    df_norm = compute_normalized(df_long)
    df_ratio = compute_ratio(df_long)
    df_prop = compute_proportions(df_long)
    
    # Compute John's proportion calculations
    df_john_prop_all = compute_john_proportions(df_orig, proportion_type='all_non_starter')
    df_john_prop_local = compute_john_proportions(df_orig, proportion_type='local_vs_long_distance')
    
    # Raw data (no normalization) - use the original long-format data
    df_raw = df_long.copy()

    # Prepare data subsets
    is_local_norm = df_norm["Region"].astype(str).isin(LOCAL_SITE_NAMES)
    df_local_norm = df_norm[is_local_norm & df_norm["Layer"].notna()].copy()
    df_long_norm = df_norm[~df_norm["Region"].astype(str).isin(LOCAL_SITE_NAMES)].copy()
    
    is_local_ratio = df_ratio["Region"].astype(str).isin(LOCAL_SITE_NAMES)
    df_local_ratio = df_ratio[is_local_ratio & df_ratio["Layer"].notna()].copy()
    df_long_ratio = df_ratio[~df_ratio["Region"].astype(str).isin(LOCAL_SITE_NAMES)].copy()
    
    is_local_prop = df_prop["Region"].astype(str).isin(LOCAL_SITE_NAMES)
    df_local_prop = df_prop[is_local_prop & df_prop["Layer"].notna()].copy()
    df_long_prop = df_prop[~df_prop["Region"].astype(str).isin(LOCAL_SITE_NAMES)].copy()
    
    is_local_raw = df_raw["Region"].astype(str).isin(LOCAL_SITE_NAMES)
    df_local_raw = df_raw[is_local_raw & df_raw["Layer"].notna()].copy()
    df_long_raw = df_raw[~df_raw["Region"].astype(str).isin(LOCAL_SITE_NAMES)].copy()

    # Run statistical tests for normalized, ratio, and proportion data
    stats_region_norm = run_per_stratum_tests(df_long_norm, conds, "Region", "Norm") if not df_long_norm.empty else pd.DataFrame()
    stats_layer_norm = run_per_stratum_tests(df_local_norm, conds, "Layer", "Norm") if not df_local_norm.empty else pd.DataFrame()
    stats_region_ratio = run_per_stratum_tests(df_long_ratio, conds, "Region", "Ratio") if not df_long_ratio.empty else pd.DataFrame()
    stats_layer_ratio = run_per_stratum_tests(df_local_ratio, conds, "Layer", "Ratio") if not df_local_ratio.empty else pd.DataFrame()
    stats_region_prop = run_per_stratum_tests(df_long_prop, conds, "Region", "Proportion") if not df_long_prop.empty else pd.DataFrame()
    stats_layer_prop = run_per_stratum_tests(df_local_prop, conds, "Layer", "Proportion") if not df_local_prop.empty else pd.DataFrame()
    stats_region_raw = run_per_stratum_tests(df_long_raw, conds, "Region", "Cells") if not df_long_raw.empty else pd.DataFrame()
    stats_layer_raw = run_per_stratum_tests(df_local_raw, conds, "Layer", "Cells") if not df_local_raw.empty else pd.DataFrame()

    # Sanity check: Force all Mann-Whitney U tests
    stats_region_norm_mw = run_per_stratum_tests_forced_mw(df_long_norm, conds, "Region", "Norm") if not df_long_norm.empty else pd.DataFrame()
    stats_layer_norm_mw = run_per_stratum_tests_forced_mw(df_local_norm, conds, "Layer", "Norm") if not df_local_norm.empty else pd.DataFrame()
    stats_region_ratio_mw = run_per_stratum_tests_forced_mw(df_long_ratio, conds, "Region", "Ratio") if not df_long_ratio.empty else pd.DataFrame()
    stats_layer_ratio_mw = run_per_stratum_tests_forced_mw(df_local_ratio, conds, "Layer", "Ratio") if not df_local_ratio.empty else pd.DataFrame()
    stats_region_prop_mw = run_per_stratum_tests_forced_mw(df_long_prop, conds, "Region", "Proportion") if not df_long_prop.empty else pd.DataFrame()
    stats_layer_prop_mw = run_per_stratum_tests_forced_mw(df_local_prop, conds, "Layer", "Proportion") if not df_local_prop.empty else pd.DataFrame()
    stats_region_raw_mw = run_per_stratum_tests_forced_mw(df_long_raw, conds, "Region", "Cells") if not df_long_raw.empty else pd.DataFrame()
    stats_layer_raw_mw = run_per_stratum_tests_forced_mw(df_local_raw, conds, "Layer", "Cells") if not df_local_raw.empty else pd.DataFrame()

    # Sanity check: Force all Student's t-tests
    stats_region_norm_st = run_per_stratum_tests_forced_st(df_long_norm, conds, "Region", "Norm") if not df_long_norm.empty else pd.DataFrame()
    stats_layer_norm_st = run_per_stratum_tests_forced_st(df_local_norm, conds, "Layer", "Norm") if not df_local_norm.empty else pd.DataFrame()
    stats_region_ratio_st = run_per_stratum_tests_forced_st(df_long_ratio, conds, "Region", "Ratio") if not df_long_ratio.empty else pd.DataFrame()
    stats_layer_ratio_st = run_per_stratum_tests_forced_st(df_local_ratio, conds, "Layer", "Ratio") if not df_local_ratio.empty else pd.DataFrame()
    stats_region_prop_st = run_per_stratum_tests_forced_st(df_long_prop, conds, "Region", "Proportion") if not df_long_prop.empty else pd.DataFrame()
    stats_layer_prop_st = run_per_stratum_tests_forced_st(df_local_prop, conds, "Layer", "Proportion") if not df_local_prop.empty else pd.DataFrame()
    stats_region_raw_st = run_per_stratum_tests_forced_st(df_long_raw, conds, "Region", "Cells") if not df_long_raw.empty else pd.DataFrame()
    stats_layer_raw_st = run_per_stratum_tests_forced_st(df_local_raw, conds, "Layer", "Cells") if not df_local_raw.empty else pd.DataFrame()

    stats_all_norm = pd.concat([stats_region_norm, stats_layer_norm], ignore_index=True)
    stats_all_ratio = pd.concat([stats_region_ratio, stats_layer_ratio], ignore_index=True)
    stats_all_prop = pd.concat([stats_region_prop, stats_layer_prop], ignore_index=True)
    stats_all_raw = pd.concat([stats_region_raw, stats_layer_raw], ignore_index=True)
    stats_all_norm_mw = pd.concat([stats_region_norm_mw, stats_layer_norm_mw], ignore_index=True)
    stats_all_ratio_mw = pd.concat([stats_region_ratio_mw, stats_layer_ratio_mw], ignore_index=True)
    stats_all_prop_mw = pd.concat([stats_region_prop_mw, stats_layer_prop_mw], ignore_index=True)
    stats_all_raw_mw = pd.concat([stats_region_raw_mw, stats_layer_raw_mw], ignore_index=True)
    stats_all_norm_st = pd.concat([stats_region_norm_st, stats_layer_norm_st], ignore_index=True)
    stats_all_ratio_st = pd.concat([stats_region_ratio_st, stats_layer_ratio_st], ignore_index=True)
    stats_all_prop_st = pd.concat([stats_region_prop_st, stats_layer_prop_st], ignore_index=True)
    stats_all_raw_st = pd.concat([stats_region_raw_st, stats_layer_raw_st], ignore_index=True)
    
    # Save statistics
    stats_csv_norm = f"{out_prefix}_stats_normalized.csv"
    stats_csv_ratio = f"{out_prefix}_stats_ratio.csv"
    stats_csv_prop = f"{out_prefix}_stats_proportion.csv"
    stats_csv_raw = f"{out_prefix}_stats_RAW.csv"
    stats_csv_norm_mw = f"{out_prefix}_stats_normalized_MW.csv"
    stats_csv_ratio_mw = f"{out_prefix}_stats_ratio_MW.csv"
    stats_csv_prop_mw = f"{out_prefix}_stats_proportion_MW.csv"
    stats_csv_raw_mw = f"{out_prefix}_stats_RAW_MW.csv"
    stats_csv_norm_st = f"{out_prefix}_stats_normalized_ST.csv"
    stats_csv_ratio_st = f"{out_prefix}_stats_ratio_ST.csv"
    stats_csv_prop_st = f"{out_prefix}_stats_proportion_ST.csv"
    stats_csv_raw_st = f"{out_prefix}_stats_RAW_ST.csv"
    
    stats_all_norm.to_csv(stats_csv_norm, index=False)
    stats_all_ratio.to_csv(stats_csv_ratio, index=False)
    stats_all_prop.to_csv(stats_csv_prop, index=False)
    stats_all_raw.to_csv(stats_csv_raw, index=False)
    stats_all_norm_mw.to_csv(stats_csv_norm_mw, index=False)
    stats_all_ratio_mw.to_csv(stats_csv_ratio_mw, index=False)
    stats_all_prop_mw.to_csv(stats_csv_prop_mw, index=False)
    stats_all_raw_mw.to_csv(stats_csv_raw_mw, index=False)
    stats_all_norm_st.to_csv(stats_csv_norm_st, index=False)
    stats_all_ratio_st.to_csv(stats_csv_ratio_st, index=False)
    stats_all_prop_st.to_csv(stats_csv_prop_st, index=False)
    stats_all_raw_st.to_csv(stats_csv_raw_st, index=False)

    # Generate plots
    fig_regions_norm = f"{out_prefix}_regions_normalized.png"
    fig_layers_norm = f"{out_prefix}_layers_normalized.png"
    fig_regions_ratio = f"{out_prefix}_regions_ratio.png"
    fig_layers_ratio = f"{out_prefix}_layers_ratio.png"
    fig_regions_prop = f"{out_prefix}_regions_proportion.png"
    fig_layers_prop = f"{out_prefix}_layers_proportion.png"
    fig_regions_raw = f"{out_prefix}_regions_RAW.png"
    fig_layers_raw = f"{out_prefix}_layers_RAW.png"
    fig_significance_norm = f"{out_prefix}_significance_effect_normalized.png"
    fig_significance_ratio = f"{out_prefix}_significance_effect_ratio.png"
    fig_significance_prop = f"{out_prefix}_significance_effect_proportion.png"
    fig_significance_raw = f"{out_prefix}_significance_effect_RAW.png"
    
    # John's proportion plots
    fig_john_prop_all = f"{out_prefix}_john_proportion_all_non_starter.png"
    fig_john_prop_local = f"{out_prefix}_john_proportion_local_vs_long_distance.png"
    fig_john_regional = f"{out_prefix}_john_proportion_regional.png"
    fig_john_regional_area = f"{out_prefix}_john_proportion_regional_area_normalized.png"
    
    # MW sanity check plots
    fig_regions_norm_mw = f"{out_prefix}_regions_normalized_MW.png"
    fig_layers_norm_mw = f"{out_prefix}_layers_normalized_MW.png"
    fig_regions_ratio_mw = f"{out_prefix}_regions_ratio_MW.png"
    fig_layers_ratio_mw = f"{out_prefix}_layers_ratio_MW.png"
    fig_regions_prop_mw = f"{out_prefix}_regions_proportion_MW.png"
    fig_layers_prop_mw = f"{out_prefix}_layers_proportion_MW.png"
    fig_significance_norm_mw = f"{out_prefix}_significance_effect_normalized_MW.png"
    fig_significance_ratio_mw = f"{out_prefix}_significance_effect_ratio_MW.png"
    fig_significance_prop_mw = f"{out_prefix}_significance_effect_proportion_MW.png"
    
    # ST sanity check plots
    fig_regions_norm_st = f"{out_prefix}_regions_normalized_ST.png"
    fig_layers_norm_st = f"{out_prefix}_layers_normalized_ST.png"
    fig_regions_ratio_st = f"{out_prefix}_regions_ratio_ST.png"
    fig_layers_ratio_st = f"{out_prefix}_layers_ratio_ST.png"
    fig_regions_prop_st = f"{out_prefix}_regions_proportion_ST.png"
    fig_layers_prop_st = f"{out_prefix}_layers_proportion_ST.png"
    fig_significance_norm_st = f"{out_prefix}_significance_effect_normalized_ST.png"
    fig_significance_ratio_st = f"{out_prefix}_significance_effect_ratio_ST.png"
    fig_significance_prop_st = f"{out_prefix}_significance_effect_proportion_ST.png"
    
    # Normalized plots
    if not df_long_norm.empty:
        plot_long_distance(df_long_norm, stats_region_norm, fig_regions_norm)
    if not df_local_norm.empty:
        plot_layers(df_local_norm, stats_layer_norm, fig_layers_norm)
    plot_significance_effect_size(stats_all_norm, fig_significance_norm)
    
    # Ratio plots
    if not df_long_ratio.empty:
        plot_long_distance_ratio(df_long_ratio, stats_region_ratio, fig_regions_ratio)
    if not df_local_ratio.empty:
        plot_layers_ratio(df_local_ratio, stats_layer_ratio, fig_layers_ratio)
    plot_significance_effect_size(stats_all_ratio, fig_significance_ratio)
    
    # Proportion plots
    if not df_long_prop.empty:
        plot_long_distance_proportion(df_long_prop, stats_region_prop, fig_regions_prop)
    if not df_local_prop.empty:
        plot_layers_proportion(df_local_prop, stats_layer_prop, fig_layers_prop)
    plot_significance_effect_size(stats_all_prop, fig_significance_prop)
    
    # Raw data plots
    if not df_long_raw.empty:
        plot_long_distance_raw(df_long_raw, stats_region_raw, fig_regions_raw)
    if not df_local_raw.empty:
        plot_layers_raw(df_local_raw, stats_layer_raw, fig_layers_raw)
    plot_significance_effect_size(stats_all_raw, fig_significance_raw)
    
    # MW sanity check plots
    if not df_long_norm.empty:
        plot_long_distance(df_long_norm, stats_region_norm_mw, fig_regions_norm_mw)
    if not df_local_norm.empty:
        plot_layers(df_local_norm, stats_layer_norm_mw, fig_layers_norm_mw)
    plot_significance_effect_size(stats_all_norm_mw, fig_significance_norm_mw)
    
    if not df_long_ratio.empty:
        plot_long_distance_ratio(df_long_ratio, stats_region_ratio_mw, fig_regions_ratio_mw)
    if not df_local_ratio.empty:
        plot_layers_ratio(df_local_ratio, stats_layer_ratio_mw, fig_layers_ratio_mw)
    plot_significance_effect_size(stats_all_ratio_mw, fig_significance_ratio_mw)
    
    if not df_long_prop.empty:
        plot_long_distance_proportion(df_long_prop, stats_region_prop_mw, fig_regions_prop_mw)
    if not df_local_prop.empty:
        plot_layers_proportion(df_local_prop, stats_layer_prop_mw, fig_layers_prop_mw)
    plot_significance_effect_size(stats_all_prop_mw, fig_significance_prop_mw)
    
    # ST sanity check plots
    if not df_long_norm.empty:
        plot_long_distance(df_long_norm, stats_region_norm_st, fig_regions_norm_st)
    if not df_local_norm.empty:
        plot_layers(df_local_norm, stats_layer_norm_st, fig_layers_norm_st)
    plot_significance_effect_size(stats_all_norm_st, fig_significance_norm_st)
    
    if not df_long_ratio.empty:
        plot_long_distance_ratio(df_long_ratio, stats_region_ratio_st, fig_regions_ratio_st)
    if not df_local_ratio.empty:
        plot_layers_ratio(df_local_ratio, stats_layer_ratio_st, fig_layers_ratio_st)
    plot_significance_effect_size(stats_all_ratio_st, fig_significance_ratio_st)
    
    if not df_long_prop.empty:
        plot_long_distance_proportion(df_long_prop, stats_region_prop_st, fig_regions_prop_st)
    if not df_local_prop.empty:
        plot_layers_proportion(df_local_prop, stats_layer_prop_st, fig_layers_prop_st)
    plot_significance_effect_size(stats_all_prop_st, fig_significance_prop_st)
    
    # John's proportion plots
    print("Generating John's proportion plots...")
    plot_john_proportions_bar_chart(df_orig, proportion_type='all_non_starter', savepath=fig_john_prop_all)
    plot_john_proportions_bar_chart(df_orig, proportion_type='local_vs_long_distance', savepath=fig_john_prop_local)
    plot_john_regional_inputs_bar_chart(df_orig, normalize_by_area=False, savepath=fig_john_regional)
    plot_john_regional_inputs_bar_chart(df_orig, normalize_by_area=True, savepath=fig_john_regional_area)
    
    # Create power analysis summary Excel file
    print("Creating power analysis summary...")
    power_summary_file = f"{out_prefix}_power_analysis_summary.xlsx"
    stats_files = {
        'normalized': stats_csv_norm,
        'ratio': stats_csv_ratio,
        'proportion': stats_csv_prop,
        'RAW': stats_csv_raw
    }
    create_power_analysis_summary(stats_files, power_summary_file)
    
    # Create practical effect threshold analysis
    print("Creating practical effect threshold analysis...")
    practical_thresholds = {}
    for analysis_type, stats_file in stats_files.items():
        try:
            df_stats = pd.read_csv(stats_file)
            practical_df = calculate_practical_effect_thresholds(df_stats)
            practical_csv = f"{out_prefix}_practical_effect_thresholds_{analysis_type}.csv"
            practical_df.to_csv(practical_csv, index=False)
            practical_thresholds[analysis_type] = practical_csv
            print(f"  Created: {practical_csv}")
        except Exception as e:
            print(f"  Error creating practical thresholds for {analysis_type}: {e}")
    
    # Create combined practical thresholds summary
    practical_summary_file = f"{out_prefix}_practical_effect_summary.xlsx"
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    
    for analysis_type, practical_csv in practical_thresholds.items():
        try:
            df = pd.read_csv(practical_csv)
            ws = wb.create_sheet(f"Practical_{analysis_type.title()}")
            
            # Write data
            for r in dataframe_to_rows(df, index=False, header=True):
                ws.append(r)
            
            # Format headers
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
                cell.alignment = Alignment(horizontal="center")
            
            # Auto-adjust column widths
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width
                
        except Exception as e:
            print(f"Error creating practical summary sheet for {analysis_type}: {e}")
    
    wb.save(practical_summary_file)
    print(f"Practical effect summary saved to: {practical_summary_file}")

    # Summary statistics with improved SEM calculation
    def safe_sem(x):
        """Calculate SEM with proper handling of single data points."""
        n = x.count()
        if n <= 1:
            return np.nan
        return x.std(ddof=1) / np.sqrt(n)
    
    summary_norm = (df_norm.groupby(["Condition", "Region", "Layer"], dropna=False)["Norm"]
                     .agg(n="count", mean="mean", sem=safe_sem)
                     .reset_index())
    summary_ratio = (df_ratio.groupby(["Condition", "Region", "Layer"], dropna=False)["Ratio"]
                      .agg(n="count", mean="mean", sem=safe_sem)
                      .reset_index())
    
    summary_csv_norm = f"{out_prefix}_summary_normalized.csv"
    summary_csv_ratio = f"{out_prefix}_summary_ratio.csv"
    summary_norm.to_csv(summary_csv_norm, index=False)
    summary_ratio.to_csv(summary_csv_ratio, index=False)

    return {
        # Normalized data files
        "stats_csv_normalized": stats_csv_norm,
        "summary_csv_normalized": summary_csv_norm,
        "fig_regions_normalized": fig_regions_norm if not df_long_norm.empty else "",
        "fig_layers_normalized": fig_layers_norm if not df_local_norm.empty else "",
        "fig_significance_normalized": fig_significance_norm,
        # Ratio data files
        "stats_csv_ratio": stats_csv_ratio,
        "summary_csv_ratio": summary_csv_ratio,
        "fig_regions_ratio": fig_regions_ratio if not df_long_ratio.empty else "",
        "fig_layers_ratio": fig_layers_ratio if not df_local_ratio.empty else "",
        "fig_significance_ratio": fig_significance_ratio,
        # Proportion data files
        "stats_csv_proportion": stats_csv_prop,
        "fig_regions_proportion": fig_regions_prop if not df_long_prop.empty else "",
        "fig_layers_proportion": fig_layers_prop if not df_local_prop.empty else "",
        "fig_significance_proportion": fig_significance_prop,
        # Raw data files
        "stats_csv_RAW": stats_csv_raw,
        "fig_regions_RAW": fig_regions_raw if not df_long_raw.empty else "",
        "fig_layers_RAW": fig_layers_raw if not df_local_raw.empty else "",
        "fig_significance_RAW": fig_significance_raw,
        # MW sanity check files
        "stats_csv_normalized_MW": stats_csv_norm_mw,
        "stats_csv_ratio_MW": stats_csv_ratio_mw,
        "stats_csv_proportion_MW": stats_csv_prop_mw,
        "stats_csv_RAW_MW": stats_csv_raw_mw,
        "fig_regions_normalized_MW": fig_regions_norm_mw if not df_long_norm.empty else "",
        "fig_layers_normalized_MW": fig_layers_norm_mw if not df_local_norm.empty else "",
        "fig_significance_normalized_MW": fig_significance_norm_mw,
        "fig_regions_ratio_MW": fig_regions_ratio_mw if not df_long_ratio.empty else "",
        "fig_layers_ratio_MW": fig_layers_ratio_mw if not df_local_ratio.empty else "",
        "fig_significance_ratio_MW": fig_significance_ratio_mw,
        "fig_regions_proportion_MW": fig_regions_prop_mw if not df_long_prop.empty else "",
        "fig_layers_proportion_MW": fig_layers_prop_mw if not df_local_prop.empty else "",
        "fig_significance_proportion_MW": fig_significance_prop_mw,
        # ST sanity check files
        "stats_csv_normalized_ST": stats_csv_norm_st,
        "stats_csv_ratio_ST": stats_csv_ratio_st,
        "stats_csv_proportion_ST": stats_csv_prop_st,
        "stats_csv_RAW_ST": stats_csv_raw_st,
        "fig_regions_normalized_ST": fig_regions_norm_st if not df_long_norm.empty else "",
        "fig_layers_normalized_ST": fig_layers_norm_st if not df_local_norm.empty else "",
        "fig_significance_normalized_ST": fig_significance_norm_st,
        "fig_regions_ratio_ST": fig_regions_ratio_st if not df_long_ratio.empty else "",
        "fig_layers_ratio_ST": fig_layers_ratio_st if not df_local_ratio.empty else "",
        "fig_significance_ratio_ST": fig_significance_ratio_st,
        "fig_regions_proportion_ST": fig_regions_prop_st if not df_long_prop.empty else "",
        "fig_layers_proportion_ST": fig_layers_prop_st if not df_local_prop.empty else "",
        "fig_significance_proportion_ST": fig_significance_prop_st,
        # John's proportion plots
        "fig_john_proportion_all_non_starter": fig_john_prop_all,
        "fig_john_proportion_local_vs_long_distance": fig_john_prop_local,
        "fig_john_proportion_regional": fig_john_regional,
        "fig_john_proportion_regional_area_normalized": fig_john_regional_area,
        # Power analysis summary
        "power_analysis_summary": power_summary_file,
    }

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = analyze()
        for k, v in outputs.items():
            if v:
                print(f"Wrote: {v}")
        print("\nNote: SVG versions of all plots have also been generated (same filenames with .svg extension)")
