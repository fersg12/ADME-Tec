import sys
import os


# Add the 'scr' folder to the Python path so internal modules can be imported

from .desirability_utils import (
    gaussian_desirability,
    trapezoidal_desirability_with_floor,
    desirability_increasing,
    desirability_decreasing
)

from .uncertainty_utils import (
    uncertainty_gaussian,
    uncertainty_trapezoidal,
    uncertainty_sigmoidal
    
)

# Define desirability function configurations

desirability_families = {
    "gaussian": gaussian_desirability,
    "trapezoidal": trapezoidal_desirability_with_floor,
    "increasing": desirability_increasing,
    "decreasing": desirability_decreasing,
}

UNCERTAINTY_FUNCTIONS = {
    "ugaussian": uncertainty_gaussian,
    "utrapezoidal": uncertainty_trapezoidal,
    "usigmoidal": uncertainty_sigmoidal
}



PROPERTY_CONFIG = {

    # =========================
    # Physicochemical
    # =========================

    "molecular_weight": {
        "family": "gaussian",
        "params": {"mu": "mu_molecular_weight", "sigma": "sigma_molecular_weight"},
        "weight": "molecular_weight",
        "uncertainty": "ugaussian"
    },

    "logP": {
        "family": "trapezoidal",
        "params": {
            "a": "logP_trap[0]",
            "b": "logP_trap[1]",
            "c": "logP_trap[2]",
            "d": "logP_trap[3]",
            "floor": 0.05
        },
        "weight": "logP",
        "uncertainty": "utrapezoidal"
    },

    "hydrogen_bond_acceptors": {
        "family": "gaussian",
        "params": {"mu": "mu_hydrogen_bond_acceptors", "sigma": "sigma_hydrogen_bond_acceptors"},
        "weight": "hydrogen_bond_acceptors",
        "uncertainty": "ugaussian"
    },

    "hydrogen_bond_donors": {
        "family": "gaussian",
        "params": {"mu": "mu_hydrogen_bond_donors", "sigma": "sigma_hydrogen_bond_donors"},
        "weight": "hydrogen_bond_donors",
        "uncertainty": "ugaussian"
    },

    "tpsa": {
        "family": "trapezoidal",
        "params": {
            "a": "tpsa_trap[0]",
            "b": "tpsa_trap[1]",
            "c": "tpsa_trap[2]",
            "d": "tpsa_trap[3]",
            "floor": 0.05
        },
        "weight": "tpsa",
        "uncertainty": "utrapezoidal"
    },

    "stereo_centers": {
        "family": "gaussian",
        "params": {"mu": "mu_stereo_centers", "sigma": "sigma_stereo_centers"},
        "weight": "stereo_centers",
        "uncertainty": "ugaussian"
    },

    # =========================
    # Absorption & Distribution
    # =========================
    "Bioavailability_Ma": {
        "family": "increasing",
        "params": {"a": "a_Bioavailability_Ma", "b": "b_Bioavailability_Ma"},
        "weight": "Bioavailability_Ma",
        "uncertainty": "usigmoidal"
    },

    "HIA_Hou": {
        "family": "increasing",
        "params": {"a": "a_HIA_Hou", "b": "b_HIA_Hou"},
        "weight": "HIA_Hou",
        "uncertainty": "usigmoidal"
    },

    "BBB_Martins": {
        "family": "increasing",
        "params": {
            "a": "a_BBB_Martins",
            "b": "b_BBB_Martins"
        },
        "weight": "BBB_Martins",
        "uncertainty": "usigmoidal"
    },

    "BBB_Martins_Safe": {
        "family": "decreasing",
        "params": {
            "a": "a_BBB_Martins",
            "b": "b_BBB_Martins"
        },
        "weight": "BBB_Martins_Safe",
        "uncertainty": "usigmoidal"
    },
    
    "Caco2_Wang": {
        "family": "increasing",
        "params": {"a": "a_Caco2_Wang", "b": "b_Caco2_Wang"},
        "weight": "Caco2_Wang",
        "uncertainty": "usigmoidal"
    },

    "PAMPA_NCATS": {
        "family": "increasing",
        "params": {"a": "a_PAMPA_NCATS", "b": "b_PAMPA_NCATS"},
        "weight": "PAMPA_NCATS",
        "uncertainty": "usigmoidal"
    },

    "Pgp_Broccatelli": {
        "family": "decreasing",
        "params": {"a": "a_Pgp_Broccatelli", "b": "b_Pgp_Broccatelli"},
        "weight": "Pgp_Broccatelli",
        "uncertainty": "usigmoidal"
    },

    "VDss_Lombardo": {
        "family": "gaussian",
        "params": {"mu": "mu_VDss_Lombardo", "sigma": "sigma_VDss_Lombardo"},
        "weight": "VDss_Lombardo",
        "uncertainty": "ugaussian"
    },

    "PPBR_AZ": {
        "family": "gaussian",
        "params": {"mu": "mu_PPBR_AZ", "sigma": "sigma_PPBR_AZ"},
        "weight": "PPBR_AZ",
        "uncertainty": "ugaussian"
    },

    "HydrationFreeEnergy_FreeSolv": {
        "family": "gaussian",
        "params": {"mu": "mu_HydrationFreeEnergy_FreeSolv", "sigma": "sigma_HydrationFreeEnergy_FreeSolv"},
        "weight": "HydrationFreeEnergy_FreeSolv",
        "uncertainty": "ugaussian"
    },

    # =========================
    # Metabolism
    # =========================

    "CYP1A2_Veith": {
        "family": "decreasing",
        "params": {"a": "a_CYP1A2_Veith", "b": "b_CYP1A2_Veith"},
        "weight": "CYP1A2_Veith",
        "uncertainty": "usigmoidal"
    },

    "CYP2C19_Veith": {
        "family": "decreasing",
        "params": {"a": "a_CYP2C19_Veith", "b": "b_CYP2C19_Veith"},
        "weight": "CYP2C19_Veith",
        "uncertainty": "usigmoidal"
    },

    "CYP2C9_Veith": {
        "family": "decreasing",
        "params": {"a": "a_CYP2C9_Veith", "b": "b_CYP2C9_Veith"},
        "weight": "CYP2C9_Veith",
        "uncertainty": "usigmoidal"
    },

    "CYP2C9_Substrate_CarbonMangels": {
        "family": "decreasing",
        "params": {"a": "a_CYP2C9_Substrate_CarbonMangels", "b": "b_CYP2C9_Substrate_CarbonMangels"},
        "weight": "CYP2C9_Substrate_CarbonMangels",
        "uncertainty": "usigmoidal"
    },

    "CYP2D6_Veith": {
        "family": "decreasing",
        "params": {"a": "a_CYP2D6_Veith", "b": "b_CYP2D6_Veith"},
        "weight": "CYP2D6_Veith",
        "uncertainty": "usigmoidal"
    },

    "CYP2D6_Substrate_CarbonMangels": {
        "family": "decreasing",
        "params": {"a": "a_CYP2D6_Substrate_CarbonMangels", "b": "b_CYP2D6_Substrate_CarbonMangels"},
        "weight": "CYP2D6_Substrate_CarbonMangels",
        "uncertainty": "usigmoidal"
    },

    "CYP3A4_Veith": {
        "family": "decreasing",
        "params": {"a": "a_CYP3A4_Veith", "b": "b_CYP3A4_Veith"},
        "weight": "CYP3A4_Veith",
        "uncertainty": "usigmoidal"
    },

    "CYP3A4_Substrate_CarbonMangels": {
        "family": "decreasing",
        "params": {"a": "a_CYP3A4_Substrate_CarbonMangels", "b": "b_CYP3A4_Substrate_CarbonMangels"},
        "weight": "CYP3A4_Substrate_CarbonMangels",
        "uncertainty": "usigmoidal"
    },

    # =========================
    # Clearance & PK
    # =========================

    "Clearance_Hepatocyte_AZ": {
        "family": "decreasing",
        "params": {"a": "a_Clearance_Hepatocyte_AZ", "b": "b_Clearance_Hepatocyte_AZ"},
        "weight": "Clearance_Hepatocyte_AZ",
        "uncertainty": "usigmoidal"
    },

    "Clearance_Microsome_AZ": {
        "family": "decreasing",
        "params": {"a": "a_Clearance_Microsome_AZ", "b": "b_Clearance_Microsome_AZ"},
        "weight": "Clearance_Microsome_AZ",
        "uncertainty": "usigmoidal"
    },

    "Half_Life_Obach": {
        "family": "trapezoidal",
        "params": {
            "a": "Half_Life_Obach[0]",
            "b": "Half_Life_Obach[1]",
            "c": "Half_Life_Obach[2]",
            "d": "Half_Life_Obach[3]",
            "floor": 0.05
        },
        "weight": "tpsa",
        "uncertainty": "utrapezoidal"
    },

    "Lipophilicity_AstraZeneca": {
        "family": "gaussian",
        "params": {"mu": "mu_Lipophilicity_AstraZeneca", "sigma": "sigma_Lipophilicity_AstraZeneca"},
        "weight": "Lipophilicity_AstraZeneca",
        "uncertainty": "ugaussian"
    },

    "Solubility_AqSolDB": {
        "family": "gaussian",
        "params": {"mu": "mu_Solubility_AqSolDB", "sigma": "sigma_Solubility_AqSolDB"},
        "weight": "Solubility_AqSolDB",
        "uncertainty": "ugaussian"
    }, 
      
    # =========================
    # Toxicity
    # =========================

    "AMES": {
        "family": "decreasing",
        "params": {"a": "a_AMES", "b": "b_AMES", "probabilistic": True},
        "weight": "AMES",
        "uncertainty": "usigmoidal"
    },

    "Carcinogens_Lagunin": {
        "family": "decreasing",
        "params": {"a": "a_Carcinogens_Lagunin", "b": "b_Carcinogens_Lagunin", "probabilistic": True},
        "weight": "Carcinogens_Lagunin",
        "uncertainty": "usigmoidal"
    },

    "ClinTox": {
        "family": "decreasing",
        "params": {"a": "a_ClinTox", "b": "b_ClinTox", "probabilistic": True},
        "weight": "ClinTox",
        "uncertainty": "usigmoidal"
    },

    "DILI": {
        "family": "decreasing",
        "params": {"a": "a_DILI", "b": "b_DILI", "probabilistic": True},
        "weight": "DILI",
        "uncertainty": "usigmoidal"
    },

    "Skin_Reaction": {
        "family": "decreasing",
        "params": {"a": "a_Skin_Reaction", "b": "b_Skin_Reaction"},
        "weight": "Skin_Reaction",
        "uncertainty": "usigmoidal"
    },

    "hERG": {
        "family": "decreasing",
        "params": {"a": "a_hERG", "b": "b_hERG"},
        "weight": "hERG",
        "uncertainty": "usigmoidal"
    },

    "LD50_Zhu": {
        "family": "increasing",
        "params": {"a": "a_LD50_Zhu", "b": "b_LD50_Zhu"},
        "weight": "LD50_Zhu",
        "uncertainty": "usigmoidal"
    },
    
    # =========================
    # Medicinal Chemistry
    # =========================
    
    "Lipinski": {
        "family": "increasing",
        "params": {"a": "a_Lipinski", "b": "b_Lipinski", "probabilistic": True},
        "weight": "Lipinski",
        "uncertainty": "usigmoidal"
    },
    
    "QED": {
        "family": "increasing",
        "params": {"a": "a_QED", "b": "b_QED"},
        "weight": "QED",
        "uncertainty": "usigmoidal"
    },
    # =========================
    # Tox21 – Nuclear Receptors
    # =========================

    "NR-AR-LBD": {
        "family": "decreasing",
        "params": {"a": "a_NR-AR-LBD", "b": "b_NR-AR-LBD"},
        "weight": "NR-AR-LBD",
        "uncertainty": "usigmoidal"
    },

    "NR-AR": {
        "family": "decreasing",
        "params": {"a": "a_NR-AR", "b": "b_NR-AR"},
        "weight": "NR-AR",
        "uncertainty": "usigmoidal"
    },

    "NR-AhR": {
        "family": "decreasing",
        "params": {"a": "a_NR-AhR", "b": "b_NR-AhR"},
        "weight": "NR-AhR",
        "uncertainty": "usigmoidal"
    },

    "NR-Aromatase": {
        "family": "decreasing",
        "params": {"a": "a_NR-Aromatase", "b": "b_NR-Aromatase"},
        "weight": "NR-Aromatase",
        "uncertainty": "usigmoidal"
    },

    "NR-ER-LBD": {
        "family": "decreasing",
        "params": {"a": "a_NR-ER-LBD", "b": "b_NR-ER-LBD"},
        "weight": "NR-ER-LBD",
        "uncertainty": "usigmoidal"
    },

    "NR-ER": {
        "family": "decreasing",
        "params": {"a": "a_NR-ER", "b": "b_NR-ER"},
        "weight": "NR-ER",
        "uncertainty": "usigmoidal"
    },

    "NR-PPAR-gamma": {
        "family": "decreasing",
        "params": {"a": "a_NR-PPAR-gamma", "b": "b_NR-PPAR-gamma"},
        "weight": "NR-PPAR-gamma",
        "uncertainty": "usigmoidal"
    },

    # =========================
    # Tox21 – Stress Response
    # =========================

    "SR-ARE": {
        "family": "decreasing",
        "params": {"a": "a_SR-ARE", "b": "b_SR-ARE"},
        "weight": "SR-ARE",
        "uncertainty": "usigmoidal"
    },

    "SR-ATAD5": {
        "family": "decreasing",
        "params": {"a": "a_SR-ATAD5", "b": "b_SR-ATAD5"},
        "weight": "SR-ATAD5",
        "uncertainty": "usigmoidal"
    },

    "SR-HSE": {
        "family": "decreasing",
        "params": {"a": "a_SR-HSE", "b": "b_SR-HSE"},
        "weight": "SR-HSE",
        "uncertainty": "usigmoidal"
    },

    "SR-MMP": {
        "family": "decreasing",
        "params": {"a": "a_SR-MMP", "b": "b_SR-MMP"},
        "weight": "SR-MMP",
        "uncertainty": "usigmoidal"
    },

    "SR-p53": {
        "family": "decreasing",
        "params": {"a": "a_SR-p53", "b": "b_SR-p53"},
        "weight": "SR-p53",
        "uncertainty": "usigmoidal"
    }

}




