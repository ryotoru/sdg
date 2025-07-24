from histogram import plot_numeric_columns
from windsorp import save_all_plots_to_pdf_grid, save_winsorized_df 
import os
import pandas as pd

def integrate_utilities_with_grid_pdf(
    df: pd.DataFrame,
    out_dir: str,
    winsor_pct,
    max_bins: int = 30,
    save_pngs: bool = True,
    save_pdf: bool = True,
    plots_per_page: int = 25,
    output_formats: list = None,
    label_fontsize: int = 12
):
    if save_pngs:
        if output_formats is None:
            output_formats = ["png"]
        plot_numeric_columns(
            df,
            out_dir=out_dir,
            max_bins=max_bins,
            winsor_pct=winsor_pct,
            show_progress=True,
            dpi=120,
            figsize=(8, 5),
            hist_color="skyblue",
            kde_color="darkred",
            output_formats=output_formats,
            show_kde=True,
            label_fontsize=label_fontsize
        )

    if save_pdf:
        save_all_plots_to_pdf_grid(out_dir, "all_plots.pdf", plots_per_page=plots_per_page)

    if (isinstance(winsor_pct, (tuple, list)) and any(w > 0 for w in winsor_pct)) or \
       (isinstance(winsor_pct, (int, float)) and winsor_pct > 0):
        win_str = f"{winsor_pct[0]}_{winsor_pct[1]}" if isinstance(winsor_pct, (tuple, list)) else str(winsor_pct)
        csv_path = os.path.join(out_dir, f"winsorized_{win_str}.csv")
        save_winsorized_df(df, winsor_pct=winsor_pct, output_path=csv_path)
    else:
        print("Winsorisation is 0%, skipping CSV export.")

