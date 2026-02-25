
def compute_trapezoidal_ranges(series, margin=0.05):
    a, b, c, d = series.quantile([0.05, 0.25, 0.75, 0.95])
    r = d - a
    return a - margin * r, b, c, d + margin * r


def compute_ab_values(reference_df, column, lower=5, upper=95):
    a = reference_df[column].quantile(lower / 100)
    b = reference_df[column].quantile(upper / 100)
    return a, b


def prepare_ranges_from_reference(ref_df, config):

    ranges = {}

    for prop, cfg in config.items():

        if prop not in ref_df.columns:
            continue

        family = cfg["family"]

        # --- Gaussian ---
        if family == "gaussian":
            ranges[f"mu_{prop}"] = ref_df[prop].mean()
            ranges[f"sigma_{prop}"] = ref_df[prop].std()

        # --- Trapezoidal ---
        elif family == "trapezoidal":
            ranges[f"{prop}_trap"] = compute_trapezoidal_ranges(ref_df[prop])

        # --- Sigmoidal ---
        elif family in ["increasing", "decreasing"]:
            a, b = compute_ab_values(ref_df, prop)
            ranges[f"a_{prop}"] = a
            ranges[f"b_{prop}"] = b

    return ranges