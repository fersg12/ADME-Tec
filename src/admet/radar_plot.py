import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.admet.adme_mappings import map_columns_perc


def plot_radar_with_min_max_df(
    min_df,
    max_df,
    compuestos_df,
    title
):

    # ==================================================
    # CONFIG
    # ==================================================
    REVERSE_PATTERNS = {
        "bbb": "BBB_Safe",
        "herg": "Non_hERG",
        "dili": "Non_DILI",
        "substrate": "Non_Substrate",
        "clearance": "Low_Clearance",
    }

    inverse_map = {
        v: k for k, v in map_columns_perc.items()
    }

    # ==================================================
    # COPY
    # ==================================================
    min_df = min_df.copy()
    max_df = max_df.copy()
    compuestos_df = compuestos_df.copy()

    # ==================================================
    # TRANSFORM
    # ==================================================
    def transform_df(df):

        new_df = pd.DataFrame(index=df.index)

        for col in df.columns:

            col_lower = col.lower()
            replaced = False

            for pattern in REVERSE_PATTERNS:

                if pattern in col_lower:
                    new_df[col] = 100 - df[col]
                    replaced = True
                    break

            if not replaced:
                new_df[col] = df[col]

        return new_df

    min_df = transform_df(min_df)
    max_df = transform_df(max_df)
    compuestos_df = transform_df(compuestos_df)

    # ==================================================
    # COMMON COLUMNS
    # ==================================================
    propiedades_claves = [
        col for col in compuestos_df.columns
        if col in min_df.columns and col in max_df.columns
    ]

    # ==================================================
    # LABELS
    # ==================================================
    def get_readable_name(col):

        base_name = inverse_map.get(col, col)
        name_lower = base_name.lower()

        if "bbb" in name_lower:
            return "BBB Safe"

        elif "herg" in name_lower:
            return "Non-hERG"

        elif "dili" in name_lower:
            return "Non-DILI"

        elif "clearance" in name_lower:
            return "Low Clearance"

        elif "substrate" in name_lower:
            return "Non-" + base_name.replace(
                " Substrate",
                ""
            )

        else:
            return base_name

    propiedades_legibles = [
        get_readable_name(col)
        for col in propiedades_claves
    ]

    # ==================================================
    # VALUES
    # ==================================================
    minimos = np.array(
        min_df[propiedades_claves]
        .iloc[0]
        .astype(float)
        .tolist()
    )

    maximos = np.array(
        max_df[propiedades_claves]
        .iloc[0]
        .astype(float)
        .tolist()
    )

    # ==================================================
    # FIX MIN/MAX
    # ==================================================
    mins = np.minimum(minimos, maximos)
    maxs = np.maximum(minimos, maximos)

    # ==================================================
    # ANGLES
    # ==================================================
    N = len(propiedades_claves)

    angles = np.linspace(
        0,
        2 * np.pi,
        N,
        endpoint=False
    )

    # cerrar polígonos
    angles_closed = np.concatenate([
        angles,
        [angles[0]]
    ])

    mins_closed = np.concatenate([
        mins,
        [mins[0]]
    ])

    maxs_closed = np.concatenate([
        maxs,
        [maxs[0]]
    ])

    # ==================================================
    # FIGURE
    # ==================================================
    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw=dict(polar=True)
    )

    # ==================================================
    # AXIS
    # ==================================================
    ax.set_xticks(angles)

    ax.set_xticklabels(
        propiedades_legibles,
        fontsize=11
    )

    ax.set_ylim(0, 100)

    ax.tick_params(
        axis='x',
        pad=15
    )

    # ==================================================
    # LABEL ALIGNMENT
    # ==================================================
    for label, angle in zip(
        ax.get_xticklabels(),
        angles
    ):

        if np.pi / 2 <= angle <= 3 * np.pi / 2:
            label.set_horizontalalignment('right')

        else:
            label.set_horizontalalignment('left')

    # ==================================================
    # MIN-MAX BAND
    # ==================================================
    angles_band = np.concatenate([
        angles_closed,
        angles_closed[::-1]
    ])

    values_band = np.concatenate([
        maxs_closed,
        mins_closed[::-1]
    ])

    ax.fill(
        angles_band,
        values_band,
        color='skyblue',
        alpha=0.35,
        label='Min-Max Range'
    )

    # ==================================================
    # BORDERS
    # ==================================================
    ax.plot(
        angles_closed,
        maxs_closed,
        color='skyblue',
        linewidth=1.5,
        alpha=0.8
    )

    ax.plot(
        angles_closed,
        mins_closed,
        color='skyblue',
        linewidth=1.5,
        alpha=0.8
    )

    # ==================================================
    # COMPOUNDS
    # ==================================================
    for i, (_, row) in enumerate(
        compuestos_df.iterrows()
    ):

        valores = np.array(
            row[propiedades_claves]
            .astype(float)
            .tolist()
        )

        valores_closed = np.concatenate([
            valores,
            [valores[0]]
        ])

        ax.plot(
            angles_closed,
            valores_closed,
            linewidth=3,
            label=f"Compound {i+1}"
        )

        ax.scatter(
            angles_closed,
            valores_closed,
            s=50,
            zorder=10
        )

    # ==================================================
    # GRID
    # ==================================================
    ax.grid(alpha=0.4)

    # ==================================================
    # TITLE
    # ==================================================
    plt.title(
        title,
        fontsize=22,
        y=1.08
    )

    # ==================================================
    # LEGEND
    # ==================================================
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=11
    )

    plt.tight_layout()

    return fig



def plot_desirability_heatmap(df_des):

    # Seleccionar columnas que empiezan con "desirability"
    desirability_cols = [col for col in df_des.columns if col.startswith("desirability")]

    # Asegurar que existe columna ID (ajusta si se llama diferente)
    id_col = "ID" if "ID" in df_des.columns else df_des.columns[0]

    # Subset del dataframe
    df_heatmap = df_des[[id_col] + desirability_cols].copy()

    # Usar ID como índice
    df_heatmap.set_index(id_col, inplace=True)

    return df_heatmap

def render_heatmap(df_heatmap, title="Desirability Heatmap"):

    n_rows = df_heatmap.shape[0]

    # Ajusta altura dinámicamente (clave para 68 moléculas)
    fig_height = max(8, n_rows * 0.25)

    fig, ax = plt.subplots(figsize=(10, fig_height))

    sns.heatmap(
        df_heatmap,
        cmap="RdYlGn",
        annot=False,
        vmin=0,
        vmax=1,
        linewidths=0.2,
        linecolor="gray",
        cbar_kws={"label": "Desirability Score"},
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Properties")
    ax.set_ylabel("")

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    return fig
