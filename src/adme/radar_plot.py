import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


def plot_radar_with_min_max_df(min_df, max_df, compuestos_df, title):
    """
    Generate a radar chart comparing compounds against reference minimum
    and maximum property ranges.

    Parameters:
        min_df (pd.DataFrame): DataFrame containing minimum values per property.
        max_df (pd.DataFrame): DataFrame containing maximum values per property.
        compuestos_df (pd.DataFrame): One or more compounds to plot
                                      (must share the same property columns).
        title (str): Plot title.

    Returns:
        fig (matplotlib.figure.Figure): Generated radar plot figure.
    """

    # Extract property column names from compounds DataFrame
    propiedades_claves = compuestos_df.columns.tolist()

    # Create human-readable labels (remove '_drugbank' and replace underscores)
    propiedades_legibles = [
        col.split('_drugbank')[0].replace('_', ' ')
        for col in propiedades_claves
    ]

    # Extract minimum and maximum reference values
    minimos = min_df[propiedades_claves].values[0].tolist()
    maximos = max_df[propiedades_claves].values[0].tolist()

    # Compute angular coordinates for radar chart (one per property)
    N = len(propiedades_claves)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    # Close the circular plot by repeating the first angle and values
    angles += angles[:1]
    minimos += minimos[:1]
    maximos += maximos[:1]

    # Create polar subplot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Set property labels around the circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(propiedades_legibles, fontsize=10)

    # Adjust radial label position
    ax.set_rlabel_position(0)
    ax.tick_params(pad=15)

    # Improve label alignment depending on position in the circle
    for label, angle in zip(ax.get_xticklabels(), angles):
        label.set_horizontalalignment(
            'right' if np.pi / 2 <= angle <= 3 * np.pi / 2 else 'left'
        )

    # Fill the background area up to maximum values
    ax.fill(angles, maximos, color='white', alpha=0.4)

    # Fill area between minimum and maximum values
    ax.fill_between(
        angles,
        minimos,
        maximos,
        color='skyblue',
        alpha=0.4,
        label='Min-Max Range'
    )

    # Plot each compound on the radar chart
    for idx, row in compuestos_df.iterrows():

        # Extract compound property values
        valores = row[propiedades_claves].tolist()

        # Close the circle by repeating first value
        valores += [valores[0]]

        label = f"Compound {idx+1}"

        # Plot compound line
        ax.plot(angles, valores, label=label, linewidth=2)

        # Add markers at each property point
        ax.scatter(angles, valores, s=40)

    # Add title
    plt.title(title, size=18, y=1.1)

    # Add legend below the plot
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=11
    )

    # Adjust layout spacing
    plt.tight_layout(pad=3.0)

    return fig
