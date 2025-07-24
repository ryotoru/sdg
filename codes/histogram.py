#!/usr/bin/env python3
"""
Interactive helpers for generating per‑column histogram + KDE plots with
**adaptive binning** inside a Jupyter notebook.

Key points
==========
* Robust cleaning (strip sentinel codes, configurable winsorisation).
* Automatic choice between Doane / Freedman–Diaconis / Scott **or**
  **log‑spaced** bins when the data span >500 ×.
* Hard cap on bin count (default **30**, override via `max_bins=`).
* **Edge‑case guard** — if a rule degenerates to < 3 bins while there are
  > 3 unique values (the "IQR = 0" problem), it falls back to 6 equal‑width
  bins or `max_bins`, whichever is smaller.
* **EHR-friendly**: Never skips columns, handles constant values appropriately,
  logs all processing decisions for audit trails.

Example (Notebook cell)
-----------------------
```python
import pandas as pd
from ehr_histogram_kde_auto_bins import plot_numeric_columns

# Load your EHR table
df = pd.read_csv("remove_pseuid.csv", low_memory=False)

# Make plots in the "plots" folder next to the notebook
plot_numeric_columns(df, out_dir="plots", max_bins=30, winsor_pct=0.1)
```
"""

import os
import re
import warnings
import logging
from typing import Iterable, Sequence, Set, Union, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm not available - progress bars disabled")

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    """Replace non‑filename‑safe chars with underscores."""
    # More robust sanitization
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)  # Windows forbidden chars
    sanitized = re.sub(r'[^\w\-_.]', '_', sanitized)  # Keep only alphanumeric, dash, underscore, dot
    sanitized = re.sub(r'_+', '_', sanitized)  # Collapse multiple underscores
    return sanitized.strip('_')  # Remove leading/trailing underscores


def _winsorise_series(
    s: pd.Series,
    winsor_pct: Union[float, Tuple[float, float]] = 0.1,
    sentinels: Set[float] = {888, 999, -999, -888, 777}
) -> pd.Series:
    """Return *s* after symmetric/asymmetric winsorisation & sentinel stripping.
    
    Parameters
    ----------
    s : pd.Series
        Input series to clean
    winsor_pct : float or tuple of floats
        Percentage to winsorise. If float, applied symmetrically.
        If tuple (left, right), applied asymmetrically.
    sentinels : set of float
        Sentinel values to remove
    
    Returns
    -------
    pd.Series
        Cleaned series
    """
    original_size = len(s)
    
    # Convert to numeric, coercing errors to NaN
    s_numeric = pd.to_numeric(s, errors='coerce')
    
    # Remove NaNs and sentinels
    s_clean = s_numeric.dropna()
    s_clean = s_clean[~s_clean.isin(sentinels)]
    
    if s_clean.empty:
        logger.warning(f"Series became empty after cleaning (original size: {original_size})")
        return s_clean
    
    if not winsor_pct:
        return s_clean
    
    # Handle winsorisation parameters
    if isinstance(winsor_pct, (tuple, list)):
        left, right = float(winsor_pct[0]) / 100, float(winsor_pct[1]) / 100
    else:
        left = right = float(winsor_pct) / 100
    
    # Apply winsorisation
    try:
        winsorized = stats.mstats.winsorize(s_clean, limits=(left, right))
        cleaned_size = len(winsorized)
        removed = original_size - cleaned_size
        
        if removed > 0:
            logger.info(f"Removed {removed} values ({removed/original_size*100:.1f}%) during cleaning")
            
        return pd.Series(winsorized, name=s.name)
    except Exception as e:
        logger.error(f"Winsorisation failed: {e}. Returning non-winsorized data.")
        return s_clean


def _auto_bin_edges(
    x: pd.Series, 
    max_bins: int = 30,
    winsor_pct: Union[float, Tuple[float, float]] = 0.1
) -> np.ndarray:
    """Return adaptive histogram bin *edges* for 1‑D data.
    
    Parameters
    ----------
    x : pd.Series
        Input data series
    max_bins : int
        Maximum number of bins allowed
    winsor_pct : float or tuple
        Winsorisation percentage
    
    Returns
    -------
    np.ndarray
        Array of bin edges
    """
    # Clean the data using the same process as plotting
    x_clean = _winsorise_series(x, winsor_pct)
    
    if x_clean.empty:
        logger.warning("No data left after cleaning for bin edge calculation")
        return np.array([0.0, 1.0])
    
    x_vals = x_clean.values
    
    # Check for constant values
    unique_vals = np.unique(x_vals)
    if len(unique_vals) == 1:
        # For constant values, create narrow bins around the single value
        val = unique_vals[0]
        margin = max(abs(val) * 0.001, 0.001)  # Small margin for visualization
        return np.array([val - margin, val + margin])
    
    rng = x_vals.max() - x_vals.min()
    
    # Decide on log‑space bins if appropriate (positive & >500× spread)
    if x_vals.min() > 0 and rng > 0 and (x_vals.max() / x_vals.min() > 500):
        nbins = min(int(np.sqrt(x_vals.size)), max_bins)
        logger.info(f"Using log-spaced bins ({nbins}) for wide-range data")
        return np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), nbins + 1)

    # Classical rules with better error handling
    n = x_vals.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            skew = stats.skew(x_vals)
        except Exception:
            skew = 0  # Fallback if skew calculation fails
    
    if n < 500:
        rule = "doane"
    elif abs(skew) > 1:
        rule = "fd"
    else:
        rule = "scott"

    try:
        edges = np.histogram_bin_edges(x_vals, bins=rule)
    except Exception as e:
        logger.warning(f"Bin edge calculation with rule '{rule}' failed: {e}. Using sturges rule.")
        edges = np.histogram_bin_edges(x_vals, bins='sturges')

    # Enforce max_bins cap
    if len(edges) - 1 > max_bins:
        step = int(np.ceil((len(edges) - 1) / max_bins))
        edges = edges[::step]
        # Ensure we capture the maximum value
        if edges[-1] < x_vals.max():
            edges = np.append(edges, x_vals.max())

    # ── Edge‑case fallback for very few bins ──────────────────────────────
    if (len(edges) - 1) < 3 and len(unique_vals) > 3:
        nbins = min(6, max_bins)
        edges = np.linspace(x_vals.min(), x_vals.max(), nbins + 1)
        logger.info(f"Applied fallback binning with {nbins} equal-width bins")

    return edges


# Removed _is_numeric function - we now process ALL columns


# ──────────────────────────────────────────────────────────────────────────────
# Main plotting routine
# ──────────────────────────────────────────────────────────────────────────────

def plot_numeric_columns(
    df: pd.DataFrame,
    *,
    out_dir: str = "numeric_plots",
    max_bins: int = 30,
    winsor_pct: Union[float, Tuple[float, float]] = 0.1,
    show_progress: bool = True,
    dpi: int = 120,
    figsize: Tuple[float, float] = (8, 5),
    hist_color: str = "skyblue",
    kde_color: str = "darkred",
    output_formats: Sequence[str] = ("png",),
    show_kde: bool = True,
    label_fontsize: int = 12
) -> None:
    """Generate histogram + KDE plots for every numeric column in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Source table (already loaded in memory).
    out_dir : str, default "numeric_plots"
        Folder where plots are written. Created if absent.
    max_bins : int, default 30
        Upper limit on the number of bins per histogram.
    winsor_pct : float or tuple, default 0.1
        Winsorisation percentage. Single value for symmetric, tuple for asymmetric.
    show_progress : bool, default True
        Show a progress bar if tqdm is available.
    dpi : int, default 120
        Resolution of the saved figures.
    figsize : tuple, default (8, 5)
        Figure size in inches (width, height).
    hist_color : str, default "skyblue"
        Color for histogram bars.
    kde_color : str, default "darkred"
        Color for KDE line.
    output_formats : sequence of str, default ("png",)
        File formats to save. Options: "png", "pdf", "svg".
    show_kde : bool, default True
        Whether to show KDE overlay.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Process ALL columns - let pd.to_numeric filter what's plottable
    all_cols = list(df.columns)
    
    if not all_cols:
        logger.warning("No columns found in DataFrame")
        return
    
    logger.info(f"Processing all {len(all_cols)} columns, extracting numeric data from each")
    
    # Set up progress tracking
    if show_progress and HAS_TQDM:
        iterator = tqdm(all_cols, desc="Processing columns")
    else:
        iterator = all_cols
    
    processed_count = 0
    skipped_count = 0
    
    for col in iterator:
        try:
            series = df[col]
            
            # Clean the data - this will convert to numeric and drop non-numeric
            x_clean = _winsorise_series(series, winsor_pct)
            
            if x_clean.empty:
                logger.info(f"Column '{col}': No numeric data found after conversion")
                skipped_count += 1
                continue
            
            # Get bin edges
            edges = _auto_bin_edges(series, max_bins=max_bins, winsor_pct=winsor_pct)
            
            # Calculate statistics
            miss_cnt = series.isna().sum()
            x_vals = x_clean.values
            
            # Create the plot
            plt.figure(figsize=figsize)
            
            # Regular histogram for all data (including constant values)
            plt.hist(x_vals, bins=edges, density=True, color=hist_color, 
                    edgecolor="black", alpha=0.7, label="Histogram")
            
            # KDE (guarded)
            if show_kde and len(x_vals) > 1:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        kde = stats.gaussian_kde(x_vals)
                        xx = np.linspace(x_vals.min(), x_vals.max(), 300)
                        plt.plot(xx, kde(xx), color=kde_color, lw=2, label="KDE")
                except Exception as e:
                    logger.debug(f"KDE failed for column '{col}': {e}")
             
            label_fontsize = 14  # ← Adjust as you like

            plt.ylabel("Density", fontsize=label_fontsize)
            title_suffix = f"(bins={len(edges)-1})"
            plt.title(f"Distribution of {col} {title_suffix}", fontsize=label_fontsize + 1)
            plt.xlabel(col, fontsize=label_fontsize)
            plt.grid(axis="y", alpha=0.3)
            plt.legend(fontsize=label_fontsize - 2)


            # Add missing count annotation
            if miss_cnt:
                plt.annotate(
                    f"Missing: {miss_cnt:,}", xy=(0.98, 0.98), xycoords="axes fraction",
                    ha="right", va="top", color="red", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # Add statistics text box
            unique_vals = np.unique(x_vals)
            median_val = np.median(x_vals)
            q25, q75 = np.percentile(x_vals, [25, 75])
            
            if len(unique_vals) == 1:
                stats_text = f"Constant: {unique_vals[0]:.3g}\nn = {len(x_vals):,}"
            else:
                stats_text = (
                    f"Median: {median_val:.3g}\n"
                    f"IQR: {q25:.3g}–{q75:.3g}\n"
                    f"n = {len(x_vals):,}")
                
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                     va="top", ha="left", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

            # Save in requested formats
            for fmt in output_formats:
                out_path = os.path.join(out_dir, f"{sanitize_filename(col)}.{fmt}")
                plt.tight_layout()
                plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
            
            plt.close()
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process column '{col}': {e}")
            plt.close('all')  # Ensure we don't leave open figures
            skipped_count += 1
            continue

    # Summary report
    total_cols = len(all_cols)
    logger.info(f"Processing complete: {processed_count}/{total_cols} columns had plottable numeric data")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} columns (no numeric data or errors)")
    
    print(f"Saved {processed_count} plots → {out_dir}")
    print(f"Processed {total_cols} total columns, found numeric data in {processed_count}")
    if len(output_formats) > 1:
        print(f"Formats: {', '.join(output_formats)}")


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper for quick ad‑hoc call *inside* a notebook cell
# ──────────────────────────────────────────────────────────────────────────────

def plot_csv_numeric_columns(
    csv_path: str, 
    csv_kwargs: Optional[dict] = None,
    **plot_kwargs
) -> None:
    """Load a CSV and delegate to *plot_numeric_columns*. Handy one‑liner.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    csv_kwargs : dict, optional
        Additional arguments passed to pd.read_csv()
    **plot_kwargs
        Arguments passed to plot_numeric_columns()
    """
    if csv_kwargs is None:
        csv_kwargs = {"low_memory": False}
    
    try:
        logger.info(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path, **csv_kwargs)
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        
        return plot_numeric_columns(df, **plot_kwargs)
    except Exception as e:
        logger.error(f"Failed to load or process CSV '{csv_path}': {e}")
        raise