# ADMETec

Interactive platform for **ADMET analysis, prediction, visualization, and context-aware compound prioritization** in computer-aided drug design.

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)

## What is ADMETec?

**ADMETec** is an interactive platform designed to support ADMET analysis and compound prioritization during computer-aided drug design (CADD) workflows.

The core module of ADME-Tec centralizes the evaluation of ADMET properties by integrating external experimental and reference data, including ChEMBL v36 and approved drugs, with computational predictions obtained through the APIs of **ADMET-AI** (Swanson et al., 2024) and **NERDD** (Hirte et al., 2025).

The platform provides an integrated view of molecular and pharmacokinetic properties, allowing users to evaluate candidate compounds in the context of their intended **drug discovery stage, therapeutic indication, and target**.

The resulting analysis includes:

* Relevant pharmacokinetic and physicochemical property descriptors.
* ADMET predictions covering absorption, distribution, metabolism, excretion, and toxicity.
* Reference comparisons with compounds in clinical development linked to the entered target or therapeutic indication.
* Comparison with approved drugs where applicable.
* Structural alerts based on established toxicity and medicinal-chemistry rules.
* Prediction of potential Phase I and Phase II metabolites.
* Interactive visualization of molecular and ADMET profiles.
* Context-aware prioritization of compound lists according to user-selected ADMET properties.

All information is delivered through an interactive user-accessible interface, allowing researchers to explore individual properties as well as integrated compound profiles.

## Core workflow

The ADMETec workflow combines molecular characterization, ADMET prediction, reference-based analysis, and multi-criteria compound prioritization.

### 1. Molecular input

Users can provide individual compounds or compound lists using **SMILES**. Additional information, such as a **ChEMBL ID**, **ATC code**, target, or therapeutic indication, can be used to define the biological and pharmacological context of the analysis.

### 2. Molecular and ADMET characterization

The platform calculates and presents relevant molecular and pharmacokinetic descriptors and integrates predictions obtained from external ADMET resources.

The analysis covers properties associated with:

* **Absorption**
* **Distribution**
* **Metabolism**
* **Excretion**
* **Toxicity**

Molecular features such as **Lipinski's Rule of Five, QED, stereochemistry, and structural alerts** are also evaluated.

### 3. Reference-based analysis

ADME-Tec integrates external reference information from **ChEMBL v36** and approved drugs.

When a target or therapeutic indication is specified, the platform can identify relevant compounds, including compounds in **clinical development**, and use them as a reference space for evaluating candidate molecules.

This provides a context-aware comparison between candidate compounds and molecules that have already demonstrated biological or pharmacological relevance.

### 4. Structural alerts and metabolism prediction

The platform evaluates candidate structures against established structural-alert rules associated with potential toxicity or undesirable chemical features.

In addition, potential metabolites are predicted for **Phase I and Phase II biotransformation**, providing additional information for the assessment of metabolic liabilities.

### 5. ADMET visualization

ADMET properties are presented through interactive tables and graphical representations, including **ADMET radar plots**, enabling users to rapidly compare multidimensional molecular profiles.

### 6. Context-aware compound prioritization

When processing compound lists, ADMETec enables prioritization according to the **ADMET properties selected by the user**.

The prioritization is based on a **multi-criteria desirability framework**, in which the individual desirability of each property is transformed into a normalized score and combined using user-defined weights.

Importantly, the aggregation strategy depends on the **drug discovery stage** selected by the user.

#### Hit identification

For **hit identification**, ADME-Tec uses a global desirability based on an **arithmetic weighted sum**.

The arithmetic aggregation provides greater tolerance to suboptimal values in individual properties while allowing compounds with an otherwise favorable overall profile to remain competitive.

#### Lead optimization

For **lead optimization**, ADME-Tec uses a **geometric mean desirability**.

The geometric aggregation imposes a more stringent balance among ADMET attributes. Consequently, compounds with critically low desirability in one or more key properties are penalized more strongly, reflecting the need for a more balanced profile during lead optimization.

The two aggregation strategies therefore provide different levels of stringency according to the stage of the drug discovery process.

## Who is it for?

ADME-Tec is intended for researchers working in:

* **Computer-aided drug design (CADD)**
* **Medicinal chemistry**
* **Cheminformatics**
* **Virtual screening**
* **Hit identification**
* **Lead optimization**
* **ADMET assessment**
* **Molecular property analysis**
* **Machine-learning-based drug discovery**

The platform is particularly useful when multiple candidate molecules need to be evaluated simultaneously and prioritized according to several potentially competing ADMET objectives.

## Prerequisites

Before installing ADME-Tec, the following software is recommended:

* **Python 3.10**
* **Conda** or **Miniconda**
* A compatible operating system such as Windows or Linux

The application is implemented using **Streamlit** and relies on cheminformatics and ADMET-related Python packages, including RDKit and ADMET-AI.

## Installation

ADMETec can be installed and launched using a Conda environment. There may be some differences depending on the operating system; however, the software has been tested on both **Windows and Linux**.

Clone the repository and move to the project directory:

```bash
git clone https://github.com/fersg12/ADME-Tec.git
cd ADME-Tec
```

Create the Conda environment using the provided environment file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate Adme-Tec
```

### Dependency troubleshooting

In some systems, dependency conflicts may occur during installation, particularly for ADMET-AI and related packages.

If the complete environment cannot be resolved, an alternative is to install the dependencies required by **ADMET-AI version 1.3.1**, following the requirements specified by the corresponding library release.

In addition, **svglib version 1.5.1** is required for the application.

For example:

```bash
pip install admet-ai==1.3.1
pip install svglib==1.5.1
```

## Quick start

After activating the Conda environment, launch the Streamlit application:

```bash
streamlit run ui/ADME-Tec_app.py
```

The application will start a local Streamlit server and provide a URL that can be opened in a web browser.

### Basic workflow

1. Enter one or more molecular **SMILES**.
2. Optionally provide a **ChEMBL ID**, **ATC code**, target, or therapeutic indication. If you provide ChEMBL ID select target type ("agonist", "antagonist", "inhibitors", etc)
3. Select the **drug discovery stage**.
4. Define the relevant **target location**, when applicable.
5. Click **Run ADME Analysis**.
6. Select the ADMET properties to include in the desirability score
7. Select weights for the selected proprerties. 
8. Apply context-aware prioritization to compound lists.
9. Compare candidates with relevant reference compounds.
10. Review ADMET radar plots and chemical similarity results.
11. Explore structural alerts and predicted metabolic sites.
12. Export the results for downstream analysis.

### From a CSV file

ADMETec can process lists of compounds provided as CSV files.

The input file should contain a column with molecular **smiles**. An additional compound identifier can be provided to facilitate tracking and visualization.

A typical input file can contain:

```text
ID,smiles
Compound_1,CCOc1ccc(...)
Compound_2,CCN(CC)...
Compound_3,CC1=CC=...
```

After loading the dataset, compounds can be processed through the same ADMET analysis and prioritization workflow.


## Reproducibility

ADMETec is designed to facilitate reproducible molecular analysis by maintaining the relationship between molecular input, calculated properties, prediction results, reference compounds, and prioritization settings.

For reproducibility, users should record:

* ADMETec version.
* Input dataset and molecular identifiers.
* SMILES standardization procedures.
* Drug discovery stage.
* Target and/or therapeutic indication.
* Target location.
* Selected ADMET properties.
* Reference-compound set.
* Property weights.
* Desirability aggregation method.
* Relevant software and dependency versions.

## Citation

Until a `CITATION.cff` file or archived release citation is provided, cite the software as:

```text
Saldivar-González, F. I., & Contreras-Torres, F. F. (2026).

ADMETec:Interactive platform for context-aware ADMET analysis and compound prioritization.

Version 0.1.0 [Computer software].

https://github.com/NanoBiostructuresRG-lab/chemvault
```

For reproducibility, cite the **specific ADME-Tec version** used in the associated methods, software, or data-availability section.

## License

ADME-Tec is licensed under the

[GNU Lesser General Public License v3.0 or later](LICENSE).

See `LICENSE`, `COPYING`, and `COPYING.LESSER`.

SPDX identifier: `LGPL-3.0-or-later`.

## Contact information

For questions, bug reports, or support, contact:

**Fernanda I. Saldivar-González**
Tecnológico de Monterrey
Email: [fer.saldivarg@tec.mx](mailto:fer.saldivarg@tec.mx)