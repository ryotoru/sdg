import pandas as pd
import os
import re


def sanitize_filename(name: str) -> str:
    """Sanitize column names to be filename-safe and readable."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)  # Windows forbidden chars
    sanitized = re.sub(r'[^\w\-_.]', '_', sanitized)  # Keep only alphanumeric, dash, underscore, dot
    sanitized = re.sub(r'_+', '_', sanitized)  # Collapse multiple underscores
    return sanitized.strip('_')


def save_column_numeric_type_summary(df: pd.DataFrame, output_path: str = "column_summary.csv") -> None:
    """
    Save a summary CSV with sanitized column names, original names, numeric type, 
    missing value count, and missing value percentage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to summarize.
    output_path : str
        Where to write the summary CSV.
    """
    summary = []
    total_rows = len(df)


    for col in df.columns:
        sanitized = sanitize_filename(col)
        # Try to convert to numeric to detect numeric behavior
        coerced = pd.to_numeric(df[col], errors='coerce')
        is_numeric = coerced.notna().sum() > 0
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        summary.append((sanitized, col, "numeric" if is_numeric else "non-numeric", missing_count,
            round(missing_pct, 2)))

    summary_df = pd.DataFrame(summary, columns=["sanitized_name", "original_name", "type","missing_count", "missing_percentage"])
    summary_df.to_csv(output_path, index=False)

    # Print summary
    num_numeric = sum(1 for _, _, t, _, _ in summary if t == "numeric")
    num_non_numeric = len(summary) - num_numeric
    print(f"Column summary written to: {output_path}")
    print(f"Total columns: {len(summary)}")
    print(f"Numeric columns: {num_numeric}")
    print(f"Non-numeric columns: {num_non_numeric}")

if __name__ == "__main__":
    # Replace this with your actual file path
    df = pd.read_csv("labs.csv", low_memory=False)
    save_column_numeric_type_summary(df, output_path="column_summary.csv")
