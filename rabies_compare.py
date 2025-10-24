"""
Rabies tracing comparison: coinjection vs sequential injection
- Normalization: (cells / starter_cells) / area
- Note: 'area' is given in **pixels** and assumed to be a consistent unit across comparisons. For absolute values or unit conversions, rescale externally.
- Graphs: local layers and long-distance regions
- Stats: normality (Shapiro), variance (Levene), appropriate test per stratum (Student/Welch t; Mann–Whitney if non-normal)
- Outputs: figures and a summary CSV written to current directory

Data format: Wide format CSV with columns for each layer/region
- Animal ID: identifier for biological replicate
- Injection Scheme: experimental condition ('coinjected' or 'separated')
- V1 L2/3 Upper, V1 L2/3 Lower, V1 L4, V1 L5, V1 L6a, V1 L6b: local layer counts
- V2M, V2L, dLGN: long-distance region counts
- V1 Starters: number of starter cells
- Area columns: area in pixels for each region
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
    """Compute proportions where layers sum to 100% and regions sum to 100%."""
    df = df.copy()
    
    # Calculate proportions for each animal and condition
    proportions = []
    
    for (animal, condition), group in df.groupby(['SampleID', 'Condition']):
        # Calculate layer proportions (V1 layers sum to 100%)
        layer_data = group[group['Region'] == 'V1'].copy()
        if not layer_data.empty and layer_data['Cells'].notna().all():
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
        if not region_data.empty and region_data['Cells'].notna().all():
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
        if norm1 and norm2:
            equal_var = (lev_p >= 0.05) if not np.isnan(lev_p) else True
            t_stat, p = stats.ttest_ind(x, y, equal_var=equal_var)
            test_name = "Student t" if equal_var else "Welch t"
        else:
            try:
                u_stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                t_stat = u_stat
                test_name = "Mann–Whitney U"
            except Exception:
                t_stat, p, test_name = np.nan, np.nan, "NA"
        eff = hedges_g(x, y) if "t" in test_name.lower() else np.nan
        results.append(TestResult(stratum_col, str(level), n1, n2, norm1, norm2, lev_p,
                                  test_name, t_stat, p, eff))
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
        try:
            u_stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            t_stat = u_stat
            test_name = "Mann–Whitney U (forced)"
        except Exception:
            t_stat, p, test_name = np.nan, np.nan, "NA"
        eff = np.nan  # Effect size not meaningful for non-parametric tests
        results.append(TestResult(stratum_col, str(level), n1, n2, norm1, norm2, lev_p,
                                  test_name, t_stat, p, eff))
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
        try:
            t_stat, p = stats.ttest_ind(x, y, equal_var=equal_var)
            test_name = "Student t (forced)" if equal_var else "Welch t (forced)"
        except Exception:
            t_stat, p, test_name = np.nan, np.nan, "NA"
        eff = hedges_g(x, y) if "t" in test_name.lower() else np.nan
        results.append(TestResult(stratum_col, str(level), n1, n2, norm1, norm2, lev_p,
                                  test_name, t_stat, p, eff))
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
            # Add jitter to x-coordinates to spread points
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
            # Add jitter to x-coordinates to spread points
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
    ax.set_xlim(-4, 5)  # Effect size from -4 to 5 (covers all data points with padding)
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
    if x_min < -4 or x_max > 5:
        clipped_count = len(df_plot[(df_plot['effect_size'] < -4) | (df_plot['effect_size'] > 5)])
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
            # Add jitter to x-coordinates to spread points
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

    # Prepare data subsets
    is_local = df_long["Region"].astype(str).isin(LOCAL_SITE_NAMES)
    df_local_norm = df_norm[is_local & df_norm["Layer"].notna()].copy()
    df_long_norm = df_norm[~df_norm["Region"].astype(str).isin(LOCAL_SITE_NAMES)].copy()
    df_local_ratio = df_ratio[is_local & df_ratio["Layer"].notna()].copy()
    df_long_ratio = df_ratio[~df_ratio["Region"].astype(str).isin(LOCAL_SITE_NAMES)].copy()
    df_local_prop = df_prop[is_local & df_prop["Layer"].notna()].copy()
    df_long_prop = df_prop[~df_prop["Region"].astype(str).isin(LOCAL_SITE_NAMES)].copy()

    # Run statistical tests for normalized, ratio, and proportion data
    stats_region_norm = run_per_stratum_tests(df_long_norm, conds, "Region", "Norm") if not df_long_norm.empty else pd.DataFrame()
    stats_layer_norm = run_per_stratum_tests(df_local_norm, conds, "Layer", "Norm") if not df_local_norm.empty else pd.DataFrame()
    stats_region_ratio = run_per_stratum_tests(df_long_ratio, conds, "Region", "Ratio") if not df_long_ratio.empty else pd.DataFrame()
    stats_layer_ratio = run_per_stratum_tests(df_local_ratio, conds, "Layer", "Ratio") if not df_local_ratio.empty else pd.DataFrame()
    stats_region_prop = run_per_stratum_tests(df_long_prop, conds, "Region", "Proportion") if not df_long_prop.empty else pd.DataFrame()
    stats_layer_prop = run_per_stratum_tests(df_local_prop, conds, "Layer", "Proportion") if not df_local_prop.empty else pd.DataFrame()

    # Sanity check: Force all Mann-Whitney U tests
    stats_region_norm_mw = run_per_stratum_tests_forced_mw(df_long_norm, conds, "Region", "Norm") if not df_long_norm.empty else pd.DataFrame()
    stats_layer_norm_mw = run_per_stratum_tests_forced_mw(df_local_norm, conds, "Layer", "Norm") if not df_local_norm.empty else pd.DataFrame()
    stats_region_ratio_mw = run_per_stratum_tests_forced_mw(df_long_ratio, conds, "Region", "Ratio") if not df_long_ratio.empty else pd.DataFrame()
    stats_layer_ratio_mw = run_per_stratum_tests_forced_mw(df_local_ratio, conds, "Layer", "Ratio") if not df_local_ratio.empty else pd.DataFrame()
    stats_region_prop_mw = run_per_stratum_tests_forced_mw(df_long_prop, conds, "Region", "Proportion") if not df_long_prop.empty else pd.DataFrame()
    stats_layer_prop_mw = run_per_stratum_tests_forced_mw(df_local_prop, conds, "Layer", "Proportion") if not df_local_prop.empty else pd.DataFrame()

    # Sanity check: Force all Student's t-tests
    stats_region_norm_st = run_per_stratum_tests_forced_st(df_long_norm, conds, "Region", "Norm") if not df_long_norm.empty else pd.DataFrame()
    stats_layer_norm_st = run_per_stratum_tests_forced_st(df_local_norm, conds, "Layer", "Norm") if not df_local_norm.empty else pd.DataFrame()
    stats_region_ratio_st = run_per_stratum_tests_forced_st(df_long_ratio, conds, "Region", "Ratio") if not df_long_ratio.empty else pd.DataFrame()
    stats_layer_ratio_st = run_per_stratum_tests_forced_st(df_local_ratio, conds, "Layer", "Ratio") if not df_local_ratio.empty else pd.DataFrame()
    stats_region_prop_st = run_per_stratum_tests_forced_st(df_long_prop, conds, "Region", "Proportion") if not df_long_prop.empty else pd.DataFrame()
    stats_layer_prop_st = run_per_stratum_tests_forced_st(df_local_prop, conds, "Layer", "Proportion") if not df_local_prop.empty else pd.DataFrame()

    stats_all_norm = pd.concat([stats_region_norm, stats_layer_norm], ignore_index=True)
    stats_all_ratio = pd.concat([stats_region_ratio, stats_layer_ratio], ignore_index=True)
    stats_all_prop = pd.concat([stats_region_prop, stats_layer_prop], ignore_index=True)
    stats_all_norm_mw = pd.concat([stats_region_norm_mw, stats_layer_norm_mw], ignore_index=True)
    stats_all_ratio_mw = pd.concat([stats_region_ratio_mw, stats_layer_ratio_mw], ignore_index=True)
    stats_all_prop_mw = pd.concat([stats_region_prop_mw, stats_layer_prop_mw], ignore_index=True)
    stats_all_norm_st = pd.concat([stats_region_norm_st, stats_layer_norm_st], ignore_index=True)
    stats_all_ratio_st = pd.concat([stats_region_ratio_st, stats_layer_ratio_st], ignore_index=True)
    stats_all_prop_st = pd.concat([stats_region_prop_st, stats_layer_prop_st], ignore_index=True)
    
    # Save statistics
    stats_csv_norm = f"{out_prefix}_stats_normalized.csv"
    stats_csv_ratio = f"{out_prefix}_stats_ratio.csv"
    stats_csv_prop = f"{out_prefix}_stats_proportion.csv"
    stats_csv_norm_mw = f"{out_prefix}_stats_normalized_MW.csv"
    stats_csv_ratio_mw = f"{out_prefix}_stats_ratio_MW.csv"
    stats_csv_prop_mw = f"{out_prefix}_stats_proportion_MW.csv"
    stats_csv_norm_st = f"{out_prefix}_stats_normalized_ST.csv"
    stats_csv_ratio_st = f"{out_prefix}_stats_ratio_ST.csv"
    stats_csv_prop_st = f"{out_prefix}_stats_proportion_ST.csv"
    
    stats_all_norm.to_csv(stats_csv_norm, index=False)
    stats_all_ratio.to_csv(stats_csv_ratio, index=False)
    stats_all_prop.to_csv(stats_csv_prop, index=False)
    stats_all_norm_mw.to_csv(stats_csv_norm_mw, index=False)
    stats_all_ratio_mw.to_csv(stats_csv_ratio_mw, index=False)
    stats_all_prop_mw.to_csv(stats_csv_prop_mw, index=False)
    stats_all_norm_st.to_csv(stats_csv_norm_st, index=False)
    stats_all_ratio_st.to_csv(stats_csv_ratio_st, index=False)
    stats_all_prop_st.to_csv(stats_csv_prop_st, index=False)

    # Generate plots
    fig_regions_norm = f"{out_prefix}_regions_normalized.png"
    fig_layers_norm = f"{out_prefix}_layers_normalized.png"
    fig_regions_ratio = f"{out_prefix}_regions_ratio.png"
    fig_layers_ratio = f"{out_prefix}_layers_ratio.png"
    fig_regions_prop = f"{out_prefix}_regions_proportion.png"
    fig_layers_prop = f"{out_prefix}_layers_proportion.png"
    fig_significance_norm = f"{out_prefix}_significance_effect_normalized.png"
    fig_significance_ratio = f"{out_prefix}_significance_effect_ratio.png"
    fig_significance_prop = f"{out_prefix}_significance_effect_proportion.png"
    
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

    # Summary statistics
    summary_norm = (df_norm.groupby(["Condition", "Region", "Layer"], dropna=False)["Norm"]
                     .agg(n="count", mean="mean", sem=lambda x: x.std(ddof=1)/np.sqrt(max(1, x.count())))
                     .reset_index())
    summary_ratio = (df_ratio.groupby(["Condition", "Region", "Layer"], dropna=False)["Ratio"]
                      .agg(n="count", mean="mean", sem=lambda x: x.std(ddof=1)/np.sqrt(max(1, x.count())))
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
        # MW sanity check files
        "stats_csv_normalized_MW": stats_csv_norm_mw,
        "stats_csv_ratio_MW": stats_csv_ratio_mw,
        "stats_csv_proportion_MW": stats_csv_prop_mw,
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
        "fig_regions_normalized_ST": fig_regions_norm_st if not df_long_norm.empty else "",
        "fig_layers_normalized_ST": fig_layers_norm_st if not df_local_norm.empty else "",
        "fig_significance_normalized_ST": fig_significance_norm_st,
        "fig_regions_ratio_ST": fig_regions_ratio_st if not df_long_ratio.empty else "",
        "fig_layers_ratio_ST": fig_layers_ratio_st if not df_local_ratio.empty else "",
        "fig_significance_ratio_ST": fig_significance_ratio_st,
        "fig_regions_proportion_ST": fig_regions_prop_st if not df_long_prop.empty else "",
        "fig_layers_proportion_ST": fig_layers_prop_st if not df_local_prop.empty else "",
        "fig_significance_proportion_ST": fig_significance_prop_st,
    }

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = analyze()
        for k, v in outputs.items():
            if v:
                print(f"Wrote: {v}")
