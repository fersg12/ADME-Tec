import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np


def plot_desirability_with_uncertainty(scores_df, uncertainty, identifier='identifier',
                                       top_n=None, output_path=None, show=True):
    """
    Plots desirability scores with error bars representing uncertainty.
    """
    data = scores_df.copy()
    data['Uncertainty'] = uncertainty
    data = data.sort_values('Desirability Score', ascending=False)

    if top_n is not None:
        data = data.head(top_n)

    labels = data[identifier] if identifier in data.columns else data.index.astype(str)

    plt.figure(figsize=(max(10, len(data) * 0.25), 6))
    plt.errorbar(
        x=range(len(data)),
        y=data['Desirability Score'],
        yerr=data['Uncertainty'],
        fmt='o',
        ecolor='gray',
        capsize=5,
        label='Desirability ± Uncertainty'
    )
    plt.xticks(ticks=range(len(data)), labels=labels, rotation=90, fontsize=8)
    plt.xlabel('Compound')
    plt.ylabel('Desirability Score')
    plt.title('Compound Ranking with Uncertainty (Hit Phase)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f" Plot saved to {output_path}")

    if show:
        plt.show()

    plt.close()









