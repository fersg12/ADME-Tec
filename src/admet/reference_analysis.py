import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. VARIABILITY ANALYSIS
# ============================================================

def compute_reference_variability(reference_df):
    """
    Analyze variability of ADMET properties
    in the reference set.
    """

    results = []

    # All numeric properties automatically
    numeric_df = reference_df.select_dtypes(
        include=[np.number]
    ).copy()

    # Exclude percentile-derived properties
    numeric_df = numeric_df[
        [
            c for c in numeric_df.columns
            if "_percentile" not in c.lower()
        ]
    ]

    for prop in numeric_df.columns:

        values = numeric_df[prop].dropna()

        if len(values) < 3:
            continue

        mean = values.mean()
        std = values.std()

        p10 = values.quantile(0.10)
        p25 = values.quantile(0.25)
        median = values.quantile(0.50)
        p75 = values.quantile(0.75)
        p90 = values.quantile(0.90)

        # IQR
        iqr = p75 - p25

        # CV
        if abs(mean) > 1e-8:
            cv = std / abs(mean)
        else:
            cv = np.nan

        # Normalized IQR
        if abs(median) > 1e-8:
            normalized_iqr = iqr / abs(median)
        else:
            normalized_iqr = np.nan

        # Variability category
        if pd.isna(cv):
            variability = "Undefined"

        elif cv < 0.10:
            variability = "Very low"

        elif cv < 0.20:
            variability = "Low"

        elif cv < 0.30:
            variability = "Moderate"

        elif cv < 0.50:
            variability = "High"

        else:
            variability = "Very high"

        results.append({
            "property": prop,
            "N": len(values),
            "mean": mean,
            "std": std,
            "CV": cv,
            "Variability": variability,
            "IQR": iqr,
            "normalized_IQR": normalized_iqr,
            "Reference range": f"{p10:.3f} – {p90:.3f}",
            "p10": p10,
            "p90": p90
        })

    variability_df = pd.DataFrame(results)

    # Lowest variability first
    if not variability_df.empty:
        variability_df = variability_df.sort_values(
            "normalized_IQR",
            ascending=True
        )

    return variability_df

# ============================================================
# 2. CORRELATION ANALYSIS
# ============================================================

def compute_reference_correlation(reference_df, threshold=0.80):
    """
    Computes Spearman correlation between numeric ADMET properties.
    Percentile-derived properties are excluded.
    """

    # Select numeric columns
    numeric_df = reference_df.select_dtypes(
        include=[np.number]
    ).copy()

    # Exclude percentile-derived properties
    numeric_df = numeric_df[
        [
            c for c in numeric_df.columns
            if "_percentile" not in c.lower()
            and c.lower() != "max_phase"
        ]
    ]

    # Remove columns with no variability
    numeric_df = numeric_df.loc[
        :,
        numeric_df.nunique(dropna=True) > 1
    ]

    # Spearman correlation
    corr_df = numeric_df.corr(
        method="spearman"
    )

    # Find highly correlated pairs
    high_corr_pairs = []

    columns = corr_df.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            correlation = corr_df.iloc[i, j]

            if abs(correlation) >= threshold:

                high_corr_pairs.append({
                    "Property 1": columns[i],
                    "Property 2": columns[j],
                    "Spearman correlation": correlation
                })

    high_corr_df = pd.DataFrame(
        high_corr_pairs
    )

    if not high_corr_df.empty:

        high_corr_df = high_corr_df.sort_values(
            "Spearman correlation",
            key=lambda x: abs(x),
            ascending=False
        )

    return corr_df, high_corr_df

def plot_reference_correlation(corr_df):
    """
    Creates a heatmap of the Spearman correlation matrix
    for the reference set.
    """

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    im = ax.imshow(
        corr_df.values,
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    # X-axis
    ax.set_xticks(
        range(len(corr_df.columns))
    )

    ax.set_xticklabels(
        corr_df.columns,
        rotation=90,
        fontsize=8
    )

    # Y-axis
    ax.set_yticks(
        range(len(corr_df.index))
    )

    ax.set_yticklabels(
        corr_df.index,
        fontsize=8
    )

    # Color bar
    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Spearman correlation"
    )

    ax.set_title(
        "ADMET Property Correlation"
    )

    plt.tight_layout()

    return fig