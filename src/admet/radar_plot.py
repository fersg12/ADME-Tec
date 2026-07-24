import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.admet.adme_mappings import map_columns_perc


def plot_radar_with_min_max_df(
    min_df,
    max_df,
    compuestos_df,
    selected_properties
):

    # ==================================================
    # CONFIG
    # ==================================================
    REVERSE_PATTERNS = {
        # Transport / Distribution
        "BBB safe": "BBB Safe",
        "Pgp": "Non Pgp",
        "VDss": "Low VDss",
        "PPBR": "Low PPBR",

        # CYP inhibitors
        "CYP1A2 inhibitor": "Non CYP1A2 inhibitor",
        "CYP2C19 inhibitor": "Non CYP2C19 inhibitor",
        "CYP2C9 inhibitor": "Non CYP2C9 inhibitor",
        "CYP2D6 inhibitor": "Non CYP2D6 inhibitor",
        "CYP3A4 inhibitor": "Non CYP3A4 inhibitor",

        # CYP substrates
        "CYP2C9 Substrate": "Non CYP2C9 Substrate",
        "CYP2D6 Substrate": "Non CYP2D6 Substrate",
        "CYP3A4 Substrate": "Non CYP3A4 Substrate",

        # Clearance
        "Clearance (Microsome)": "Low Clearance (Microsome)",
        "Clearance (Hepatocyte)": "Low Clearance (Hepatocyte)",

        # Toxicity
        "AMES": "Non AMES",
        "DILI": "Non DILI",
        "Carcinogenicity": "Non Carcinogenic",
        "hERG": "Non hERG",
        "Skin Reaction": "Non Skin Reaction",
        "ClinTox": "Non Clinical Toxicity",

        # Tox21
        "NR-AR-LBD": "Non NR-AR-LBD",
        "NR-AR": "Non NR-AR",
        "NR-AhR": "Non NR-AhR",
        "NR-Aromatase": "Non NR-Aromatase",
        "NR-ER-LBD": "Non NR-ER-LBD",
        "NR-ER": "Non NR-ER",
        "NR-PPAR-gamma": "Non NR-PPAR-gamma",
        "SR-ARE": "Non SR-ARE",
        "SR-ATAD5": "Non SR-ATAD5",
        "SR-HSE": "Non SR-HSE",
        "SR-MMP": "Non SR-MMP",
        "SR-p53": "Non SR-p53",
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
    
    inverse_map = {}

    for prop, col in map_columns_perc.items():

        if col == "BBB_Martins_drugbank_approved_percentile":
            continue

        inverse_map[col] = prop


    selected_properties = set(selected_properties)


    def get_property_name(col):

        if col == "BBB_Martins_drugbank_approved_percentile":

            if "BBB safe" in selected_properties:
                return "BBB safe"

            return "BBB penetration"

        return inverse_map.get(col, col)



    def transform_df(df):

        new_df = pd.DataFrame(index=df.index)

        for col in df.columns:

            property_name = get_property_name(col)

            if property_name in REVERSE_PATTERNS:
                new_df[col] = 100 - df[col]
            else:
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
    propiedades_legibles = []

    for col in propiedades_claves:

        property_name = get_property_name(col)

        propiedades_legibles.append(
            REVERSE_PATTERNS.get(property_name, property_name)
        )

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

        if "ID" in compuestos_df.columns and pd.notna(row["ID"]):
                compound_name = str(row["ID"])
        else:
                compound_name = f"Compound {i+1}"

        ax.plot(
            angles_closed,
            valores_closed,
            linewidth=3,
            label=compound_name
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

    desirability_cols = [col for col in df_des.columns if col.startswith("desirability")]
    id_col = "ID" if "ID" in df_des.columns else df_des.columns[0]
    df_heatmap = df_des[[id_col] + desirability_cols].copy()
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
