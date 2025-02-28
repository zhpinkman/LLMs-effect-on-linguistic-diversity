# LLMs Effect on Linguistic Diversity

This repository contains the code and data for the paper:

## **"The Shrinking Landscape of Linguistic Diversity in the Age of LLMs"**

## Repository Structure
The repository is organized into two main directories:

### 1. `linguistic_diversity_analysis/`
This directory contains the code and datasets for **Studies 1 and 2**, which explore the homogenization of language following the introduction of LLMs marked by ChatGPT.

#### Key Files:
- **`granger_causality_for_study_1.Rmd`** – Performs Granger causality analysis for Study 1.
- **`time_series_effect_of_chatgpt_on_writing_styles.Rmd`** – Conducts shock analysis for Study 1.
- **`data/`** – Contains all datasets used in Studies 1 and 2, including analyses on linguistic complexity variance.
  - **`analysis_of_the_variance_between_lexical_cues_agg.ipynb`** – Examines differences in variance of linguistic complexity between original and LLM-rewritten texts.
  - **`check_similarity.ipynb`** – Analyzes semantic similarity between document sets.
  - **`trends_analysis.ipynb`** – Visualizes trends in AI-generated text adoption and linguistic complexity variance.

---

### 2. `linguistic_homogenization_social_impact/`
This directory contains the datasets and code for **Studies 3 and 4**, which investigate the societal impact of linguistic homogenization caused by LLMs.

#### Key Subdirectories:
- **`datasets/`** – Stores all datasets used in Studies 3 and 4.
    - **`check_similarity.ipynb`** – Analyzes semantic similarity between document sets.
    - **`analysis_of_the_variance_between_lexical_cues_agg.ipynb`** – Investigates variance in linguistic complexity and compares original vs. LLM-rewritten texts.
    - **`analysis_of_the_variance_between_lexical_cues.ipynb`** – Studies lexical category shifts and variance (included in the appendix).
    - **`corr_analysis.ipynb`** – Conducts Study 4 analysis on associations between personal traits and lexical cues.
- **`dictionaries/`** – Contains lexical dictionaries utilized in Study 4.
- **`predictive_models/`** – Code for Study 3, which examines the effect of LLMs on the predictive power of linguistic markers for authors' personal traits.
  - **`read_results.ipynb`** – Processes and analyzes model results.
  - **`model.py`** – Code for training predictive models.
  - **`Transformer_models/`** – Implements the Longformer model, used as a Transformer model, complementary to the other classifiers used in Study 3.


Refer to `environment.yml` for the required Python packages and their versions. Installing all the requirements would take a short time (approximately 10 minutes), which is done by running the following command in the terminal:

```bash
conda env create -f environment.yml
```

Similarly, running all the scripts and notebooks that demonstrate the findings in the paper would take a few minutes, with the exception of training the classifiers using various random seeds, which would take a few hours, as well as the featurization of the datasets for Study 1 which similarly would take a few hours.

---
For any questions or access to the additional files that have been added to gitignore due to their large size, please contact the authors of the paper.
