import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

def save_all_plots_to_pdf_grid(plot_dir: str, output_pdf: str, plots_per_page: int = 25) -> None:
    """
    Combine all .png plots in a folder into a single PDF with a grid layout (e.g., 4 plots per page).

    Parameters
    ----------
    plot_dir : str
        Directory containing the PNG plot files.
    output_pdf : str
        Output PDF filename (inside plot_dir).
    plots_per_page : int
        Number of plots per page (must be a square number: 4, 9, 16, etc.).
    """
    images = sorted([f for f in os.listdir(plot_dir) if f.lower().endswith(".png")])
    pdf_path = os.path.join(plot_dir, output_pdf)

    if not images:
        print(f"No PNG images found in '{plot_dir}'.")
        return

    grid_size = int(math.sqrt(plots_per_page))
    if grid_size * grid_size != plots_per_page:
        raise ValueError("plots_per_page must be a perfect square (e.g., 4, 9, 16)")

    with PdfPages(pdf_path) as pdf:
        for i in range(0, len(images), plots_per_page):
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(11, 8.5), dpi=350)  # A4 landscape
            axs = axs.flatten()

            for j in range(plots_per_page):
                ax = axs[j]
                idx = i + j
                if idx < len(images):
                    img_path = os.path.join(plot_dir, images[idx])
                    try:
                        img = mpimg.imread(img_path)
                        ax.imshow(img)
                        ax.set_title(images[idx].rsplit('.', 1)[0], fontsize=6)

                        ax.axis("off")
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
                        ax.axis("off")
                else:
                    ax.axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved gridded PDF to: {pdf_path}")


def save_winsorized_df(
    df: pd.DataFrame,
    winsor_pct,
    sentinels={888, 999, -999, -888, 777},
    output_path="winsorized_output.csv"
) -> None:
    """
    Applies winsorisation to numeric columns and saves the cleaned DataFrame as a CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.
    winsor_pct : float or tuple
        Winsorisation percentage (symmetric or (left, right)).
    sentinels : set of float, optional
        Sentinel values to strip before winsorisation.
    output_path : str, optional
        Output path for the CSV.
    """
    def winsorize_series(s):
        s_numeric = pd.to_numeric(s, errors='coerce')
        s_clean = s_numeric.dropna()
        s_clean = s_clean[~s_clean.isin(sentinels)]

        if s_clean.empty or not winsor_pct:
            return s_clean

        if isinstance(winsor_pct, (tuple, list)):
            left, right = float(winsor_pct[0]) / 100, float(winsor_pct[1]) / 100
        else:
            left = right = float(winsor_pct) / 100

        try:
            return pd.Series(stats.mstats.winsorize(s_clean, limits=(left, right)), name=s.name)
        except Exception as e:
            print(f"Winsorisation failed for {s.name}: {e}")
            return s_clean

    numeric_cols = df.columns
    cleaned_df = df.copy()
    for col in numeric_cols:
        cleaned_series = winsorize_series(df[col])
        if not cleaned_series.empty:
            cleaned_df[col] = cleaned_series

    cleaned_df.to_csv(output_path, index=False)
    print(f"Saved winsorised DataFrame to: {output_path}")
