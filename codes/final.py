import pandas as pd
from pipeline import integrate_utilities_with_grid_pdf 

# Load your CSV
df = pd.read_csv("labs.csv")

# Run full plotting and export pipeline
integrate_utilities_with_grid_pdf(
    df=df,
    out_dir="plots",
    winsor_pct=(1, 2),
    max_bins=30,
    save_pngs=False,
    save_pdf=True,
    output_formats=["png", "svg"],
    label_fontsize=12
)
