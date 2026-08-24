
import pandas as pd
import numpy as np


TARGET_LOCATION_PROPERTIES = {

    "Crosses BBB": {

        "Absorption": {
            "logP": {
                "min": 2.0,
                "max": 4.0,
                "priority": "high"
            },
            "TPSA": {
                "max": 90,
                "priority": "high"
            },
            "Pgp": {
                "min": 0.0,
                "max": 0.5,
                "priority": "critical"
            }
        },

        "Distribution": {
            "BBB_Martins": {
                "min": 0.5,
                "max": 1.0,
                "priority": "critical"
            },
            "PPB": {
                "min": 90,
                "max": 100,
                "priority": "medium"
            },
            "molecular_weight": {
                "min": 300,
                "max": 500,
                "priority": "high"
            }
        },

        "Metabolism": {},

        "Excretion": {},

        "Toxicity": {}
    },


    "Intracellular": {

        "Absorption": {
            "logP": {
                "min": 2.0,
                "max": 5.0,
                "priority": "high"
            },
            "TPSA": {
                "max": 120,
                "priority": "high"
            },
            "permeability": {
                "min": 0.5,
                "max": 1.0,
                "priority": "critical"
            }
        },

        "Distribution": {},

        "Metabolism": {},

        "Excretion": {},

        "Toxicity": {}
    },


    "Extracellular": {

        "Absorption": {
            "Solubility": {
                "min": 0.5,
                "max": 1.0,
                "priority": "high"
            },
            "Bioavailability_Ma": {
                "min": 0.0,
                "max": 0.5,
                "priority": "critical"
            }
        },

        "Distribution": {
            "molecular_weight": {
                "max": 500,
                "priority": "medium"
            },
            "TPSA": {
                "max": 140,
                "priority": "medium"
            }
        },

        "Metabolism": {},

        "Excretion": {
            "CL": {
                "min": 0.0,
                "max": 0.5,
                "priority": "high"
            },
            "Vd": {
                "min": 0.0,
                "max": 0.5,
                "priority": "high"
            }
        },

        "Toxicity": {}
    },


    "bR05 target": {

        "Absorption": {
            "logP": {
                "min": 2.0,
                "max": 4.0,
                "priority": "high"
            },
            "passive_permeability": {
                "min": 0.0,
                "max": 0.5,
                "priority": "critical"
            }
        },

        "Distribution": {
            "molecular_weight": {
                "min": 300,
                "max": 500,
                "priority": "high"
            },
            "TPSA": {
                "min": 60,
                "max": 90,
                "priority": "high"
            },
            "PPB": {
                "min": 90,
                "max": 100,
                "priority": "medium"
            }
        },

        "Metabolism": {},

        "Excretion": {},

        "Toxicity": {}
    }
}

DISEASE_CONTEXT_PROPERTIES = {

    "Acute_Disease": {

        "Absorption": {},

        "Distribution": {},

        "Metabolism": {
            "Clearance_Hepatocyte_AZ": {
                "min": 2.0,
                "max": 5.0,
                "priority": "high"
            },
            "Clearance_Microsome_AZ": {
                "max": 120,
                "priority": "high"
            }
        },

        "Excretion": {
            "Half_Life_Obach": {
                "min": 0.5,
                "max": 1.0,
                "priority": "high"
            }
        },

        "Toxicity": {
            "LD50_Zhu": {
                "min": 0.5,
                "max": 1.0,
                "priority": "critical"
            }
        }
    },


    "Chronic_Disease": {

        "Absorption": {
            "logP": {
                "min": 2.0,
                "max": 5.0,
                "priority": "high"
            },
            "TPSA": {
                "max": 120,
                "priority": "high"
            },
            "permeability": {
                "min": 0.5,
                "max": 1.0,
                "priority": "critical"
            }
        },

        "Distribution": {},

        "Metabolism": {},

        "Excretion": {},

        "Toxicity": {}
    }
}


ADMET_CATEGORIES = [
    "Absorption",
    "Distribution",
    "Metabolism",
    "Excretion",
    "Toxicity"
]


PRIORITY_SCORE = {
    "critical": 1.00,
    "high": 0.80,
    "medium": 0.50,
    "low": 0.30
}


def recommend_properties(
    target_location,
    reference_df,
    disease_context=None,
    correlation_threshold=0.80
):

    # ============================================================
    # 1. GET TARGET LOCATION RULES
    # ============================================================

    location_rules = TARGET_LOCATION_PROPERTIES.get(
        target_location,
        {}
    )

    if not location_rules:
        return pd.DataFrame()


    # ============================================================
    # 2. START WITH EMPTY ADMET STRUCTURE
    # ============================================================

    combined_rules = {
        category: {}
        for category in ADMET_CATEGORIES
    }


    # ============================================================
    # 3. ADD TARGET LOCATION RULES
    # ============================================================

    for category in ADMET_CATEGORIES:

        category_rules = location_rules.get(
            category,
            {}
        )

        for prop, rule in category_rules.items():

            combined_rules[category][prop] = rule.copy()


    # ============================================================
    # 4. ADD DISEASE CONTEXT
    # ============================================================

    if disease_context:

        disease_rules = DISEASE_CONTEXT_PROPERTIES.get(
            disease_context,
            {}
        )

        for category in ADMET_CATEGORIES:

            category_rules = disease_rules.get(
                category,
                {}
            )

            for prop, rule in category_rules.items():

                if prop not in combined_rules[category]:

                    combined_rules[category][prop] = rule.copy()

                else:

                    current_priority = combined_rules[
                        category
                    ][prop].get(
                        "priority",
                        "medium"
                    )

                    new_priority = rule.get(
                        "priority",
                        "medium"
                    )

                    if PRIORITY_SCORE.get(
                        new_priority,
                        0
                    ) > PRIORITY_SCORE.get(
                        current_priority,
                        0
                    ):

                        combined_rules[
                            category
                        ][prop]["priority"] = new_priority


    # ============================================================
    # 5. BUILD CANDIDATE TABLE
    # ============================================================

    records = []


    for category in ADMET_CATEGORIES:

        for prop, rule in combined_rules[
            category
        ].items():

            if prop not in reference_df.columns:
                continue

            values = pd.to_numeric(
                reference_df[prop],
                errors="coerce"
            ).dropna()

            if len(values) < 2:
                continue

            mean = values.mean()
            std = values.std()

            if mean != 0:

                cv = abs(std / mean)

            else:

                cv = np.nan


            # Reference range
            reference_min = values.min()
            reference_max = values.max()


            # Expert priority
            priority = rule.get(
                "priority",
                "medium"
            )

            expert_score = PRIORITY_SCORE.get(
                priority,
                0.5
            )


            # Variability score
            if pd.isna(cv):

                variability_score = 0.0

            else:

                variability_score = min(
                    cv / 0.5,
                    1.0
                )


            # Final score
            final_score = (
                0.7 * expert_score +
                0.3 * variability_score
            )


            records.append({

                "Property": prop,

                "Category": category,

                "Priority": priority,

                "Score": final_score,

                "CV": cv,

                "Reference range":
                    f"{reference_min:.2f} – "
                    f"{reference_max:.2f}",

                "Target range":
                    rule,

                "Recommended": True

            })


    if not records:

        return pd.DataFrame()


    recommendation_df = pd.DataFrame(
        records
    )


    # ============================================================
    # 6. CORRELATION ANALYSIS
    # ============================================================

    candidates = recommendation_df[
        "Property"
    ].tolist()

    numeric_df = reference_df[
        candidates
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )


    corr = numeric_df.corr(
        method="spearman"
    )


    # ============================================================
    # 7. REMOVE REDUNDANT PROPERTIES
    # ============================================================

    recommended = set(candidates)


    for i, prop1 in enumerate(candidates):

        for prop2 in candidates[i + 1:]:

            if prop1 not in corr.columns:
                continue

            if prop2 not in corr.columns:
                continue

            rho = corr.loc[
                prop1,
                prop2
            ]

            if pd.isna(rho):
                continue

            if abs(rho) < correlation_threshold:
                continue


            row1 = recommendation_df[
                recommendation_df["Property"] == prop1
            ].iloc[0]

            row2 = recommendation_df[
                recommendation_df["Property"] == prop2
            ].iloc[0]


            priority1 = row1["Priority"]
            priority2 = row2["Priority"]


            # Critical properties are retained
            if priority1 == "critical":

                recommended.discard(prop2)

            elif priority2 == "critical":

                recommended.discard(prop1)

            else:

                score1 = row1["Score"]
                score2 = row2["Score"]

                if score1 >= score2:

                    recommended.discard(prop2)

                else:

                    recommended.discard(prop1)


    # ============================================================
    # 8. UPDATE RECOMMENDATION STATUS
    # ============================================================

    recommendation_df[
        "Recommended"
    ] = recommendation_df[
        "Property"
    ].isin(recommended)


    # ============================================================
    # 9. SORT
    # ============================================================

    recommendation_df = (
        recommendation_df
        .sort_values(
            [
                "Recommended",
                "Category",
                "Score"
            ],
            ascending=[
                False,
                True,
                False
            ]
        )
        .reset_index(drop=True)
    )


    return recommendation_df