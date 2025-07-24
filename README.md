# Automated EDA Plotting Pipeline

This project provides an automated pipeline for performing initial Exploratory Data Analysis (EDA) on tabular datasets. It generates high-quality histograms with adaptive binning, Kernel Density Estimates (KDEs), and consolidated PDF reports for all numeric columns in your data for further analysis.

## Key Features

* **Robust Data Cleaning**: Automatically handles missing values and sentinel codes, and applies Winsorisation to mitigate the effect of outliers.
* **Adaptive Histogram Binning**: chooses the correct binning strategy (Doane, Freedman-Diaconis, Scott, or log-spaced) based on the data's distribution. Can be modified as per need. 
* **Plotting**: Generates a detailed histogram and KDE plot for every numeric column in your dataset.
* **PDF Grid Reports**: Combines all generated plots into a single, easy-to-share PDF file with a grid layout.
* **Data Export**: Saves the cleaned (Winsorized) data to a new CSV file for further analysis.
* **Column Summaries**: Creates a summary file detailing the data type and missing value counts for each column.
* **EHR-Friendly**: Designed for cleaning Electronic Health Record (EHR) data.

## Structure 
**codes** all the code files. <br>
*histogram.py*: The core module for generating histograms and KDE plots with adaptive binning and data cleaning.<br>
*windsorp.py*: Contains utility functions to save plots into a gridded PDF and to save the Winsorized DataFrame to a CSV. <br>
*pipeline.py*: Integrates the plotting and utility functions into a single, configurable workflow. <br>

*final.py*: The main entry point to run the complete analysis pipeline. <br>
*column.py*: Provides a utility to generate a summary CSV of all columns, their data types, and missing value statistics. <br>

## Installation

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  Install the required dependencies:
    ```bash
    pip install pandas matplotlib scipy numpy tqdm
    ```

## Usage

The main pipeline can be run from the `final.py` script.

1.  Place your data file (e.g., `labs.csv`) in the root directory.
2.  Modify `final.py` to point to your CSV file.
3.  Run the script:
    ```bash
    python final.py
    ```

This will:
* Create a directory named `plots/`.
* Save individual PNG and SVG plots for each numeric column in the `plots/` directory.
* Generate a consolidated PDF named `all_plots.pdf` inside `plots/`.
* Save a Winsorized version of your data as `winsorized_1_2.csv` in the `plots/` directory.

### Example

The `final.py` script shows a simple example of how to use the pipeline:

```python
import pandas as pd
from pipeline import integrate_utilities_with_grid_pdf

# Load your CSV
df = pd.read_csv("labs.csv")

# Run full plotting and export pipeline
integrate_utilities_with_grid_pdf(
    df=df,
    out_dir="plots",
    winsor_pct=(1, 2), #can be set to zero if you dont want to winsorize or a singular number like 5 if want symmetric winsorization on both tails.
    max_bins=30,
    save_pngs=True, # Set to False to disable individual PNGs
    save_pdf=True,
    output_formats=["png", "svg"],
    label_fontsize=12
)
