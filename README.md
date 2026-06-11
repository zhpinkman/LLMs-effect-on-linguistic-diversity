# LLMs Effect on Linguistic Diversity

This repository accompanies the paper:

> **The Shrinking Landscape of Linguistic Diversity in the Age of Large Language Models** *(Nature Human Behaviour, 2026)*

It contains the **analysis code** (Python scripts, R Markdown notebooks, and Jupyter notebooks **with rendered outputs**) that reproduces every figure, table, and reported statistic in the paper. The underlying datasets are **not redistributed** here — see the [Data availability](#data-availability) section below for instructions on obtaining each corpus directly from its original source under the original investigators' terms.

---

## Repository structure

```
LLMs-effect-on-linguistic-diversity/
├── linguistic_diversity_analysis/                # Study 1 (arXiv / Reddit / Patch News)
│   ├── granger_causality_for_study_1.Rmd         # Granger causality + GAM nonlinear extension
│   ├── time_series_effect_of_chatgpt_on_writing_styles.Rmd  # DGM shock analysis
│   ├── utils.py
│   └── data/
│       ├── analysis_of_the_variance_between_lexical_cues_agg.ipynb
│       ├── check_similarity.ipynb                # Semantic-similarity sanity checks
│       ├── trends_analysis.ipynb                 # Fig 1 generator (with rendered outputs)
│       ├── trends_analysis_mean.ipynb            # ED Fig 5 generator
│       ├── openai_handler.py                     # OpenAI embedding / rewriting handler
│       ├── rewrite_{gpt,gemini,llama}.py         # LLM-rewriting drivers
│       └── papers/, reddit/, news/               # Per-dataset analysis CSVs (NOT redistributed)
│
└── linguistic_homogenization_social_impact/      # Studies 2 & 3 (psych corpora + classifiers)
    ├── datasets/
    │   ├── check_similarity.ipynb                # ED Fig 1 generator
    │   ├── analysis_of_the_variance_between_lexical_cues.ipynb         # ED Fig 6 generator (notebook used for the published markers)
    │   ├── analysis_of_the_variance_between_lexical_cues_agg.ipynb
    │   ├── corr_analysis.ipynb                   # Study 3 correlations / Tables F10–F15
    │   ├── create_samples_for_llm_classification_accuracy.ipynb
    │   ├── custom_posthoc.py                     # Dunn's-test helper
    │   ├── get_embeddings.py                     # OpenAI ada-002 embedding driver
    │   ├── lang_dicts.py                         # LIWC / NRC / Empathy / MFD2 loaders
    │   ├── openai_handler.py
    │   ├── rewrite_{gpt,gemini,llama}.py         # LLM-rewriting drivers
    │   ├── commands_for_{gpt,gemini,llama}_rewrite.sh  # Batched rewrite commands
    │   ├── score_dataset_wrt_dictionaries.py     # Lexical-category scoring
    │   ├── facebook/, political/, wassa/         # Per-dataset folders (NOT redistributed)
    │   └── figures/
    │
    └── predictive_models/                        # Study 2 (classifier training + reading results)
        ├── model.py                              # Classifier training entry point (SVM / LR / RF / Gradient Boosting)
        ├── read_results.ipynb                    # Reads per-run logs → Figs 2, 3, ED Figs 2, 3, 4
        ├── Transformer_models/                   # Longformer training scripts
        └── *_logs/                               # Per-(LLM × prompt × seed) prediction JSONs (NOT redistributed)
```

Notebooks are committed **with their full rendered outputs** (figures, tables, statistics), so the paper's results are visible even if you cannot re-run the underlying analyses end-to-end.

---

## Data availability

Per the data-availability statement in the manuscript, every dataset analysed here was previously collected and shared by other investigators under their own ethical approvals. The dataset files themselves are **not** redistributed in this repository (covered by `.gitignore`). Each corpus is publicly or conditionally available through the original source:

| Corpus | How to obtain | Notes |
|---|---|---|
| **Reddit r/WritingPrompts posts** *(Study 1)* | [Project Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) | Mirrors publicly accessible Reddit content. Use limited to aggregate quantitative analysis; no individual posts redistributed by this repository. |
| **arXiv abstracts** *(Study 1)* | [Kaggle arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv) (CC0 1.0 Universal Public Domain Dedication) | Filter to CS-CL and CS-CV categories per Methods §1.1. |
| **Patch.com news articles** *(Study 1)* | Public [Patch.com](https://patch.com) archive | Filtered to Jan 2018 – Nov 2023. |
| **United States Congressional Records** *(Study 2)* | [Gentzkow, Shapiro, Taddy (2018) — public parsed corpus](https://data.stanford.edu/congress_text) | Public-domain federal works. |
| **YourMorals Facebook posts + MFQ** *(Study 2)* | Released by Kennedy et al. (2021) — see paper for repository link. | Collected under USC IRB protocol UP-07-00393-AM019. |
| **Empathic Conversations / WASSA 2023** *(Study 2)* | Omitaomu et al. (2022) — see arXiv:2205.12698 and the WASSA 2023 shared-task page. | Collected under University of Pennsylvania IRB protocol #826448. |
| **Essays corpus (Pennebaker & King, 1999)** *(Study 2)* | Available **on request** from Prof. James W. Pennebaker (University of Texas at Austin; pennebaker@utexas.edu / jwpennebaker@gmail.com) | Not publicly redistributable. We obtained the corpus via direct request and were not permitted to redistribute. |
| **LLM-rewritten texts (GPT-3.5 / Gemini / Llama 3)** | Re-generate with `rewrite_{gpt,gemini,llama}.py` after providing your own API keys | Output JSONs are excluded from this repository. |

### Lexicons

| Lexicon | Status |
|---|---|
| **LIWC (LIWC-22 and LIWC-07)** | **Proprietary — not redistributed.** Obtain a licence directly from [liwc.app](https://www.liwc.app). After purchase, place the dictionary file at the path the relevant notebook/script expects (e.g. `linguistic_diversity_analysis/liwc_dictionary.dic`). |
| **NRC Emotion Lexicon** | Available from [Saif Mohammad's NRC page](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm). |
| **Empathy Lexicon (Sedoc et al., 2020)** | Available from the [LREC 2020 paper](https://aclanthology.org/2020.lrec-1.206/). |
| **Moral Foundations Dictionary 2 (MFD2)** | Available from [Frimer et al. (2019)](https://osf.io/whjt2/). |

### Source Data for figures

For convenience, the **statistical Source Data** files accompanying each main and Extended Data figure (one Excel workbook per figure) are released alongside the paper on the Nature Human Behaviour website as Source Data. These contain everything needed to verify the published figure numbers without re-running the full pipeline.

---

## Reproducing the analyses

1. **Clone this repository**, then place each acquired dataset in the path expected by the relevant notebook (`linguistic_diversity_analysis/data/{papers,reddit,news}/` for Study 1; `linguistic_homogenization_social_impact/datasets/{facebook,political,wassa,essays}/` for Studies 2–3).
2. **Install dependencies.** Python: a recent (≥ 3.10) environment with `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`, `spacy`, `nltk`, `transformers`, `openai`. R: `lme4` / `nlme`, `mgcv`, `lmtest`, `vars`, `tseries`. Software versions used in the paper are listed in Methods §1 and the Reporting Summary.
3. **Provide API keys** in your shell environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) before running any of the `rewrite_*.py` or embedding scripts.
4. **Run the notebooks / R Markdown files** in the order they appear in the paper:
   - Study 1a (Granger + DGM) → `linguistic_diversity_analysis/*.Rmd`
   - Study 1b (Levene + bootstrap) → `linguistic_diversity_analysis/data/analysis_of_the_variance_between_lexical_cues_agg.ipynb`
   - Study 2 (classifier predictive power) → `linguistic_homogenization_social_impact/predictive_models/model.py` (training) → `read_results.ipynb` (reading)
   - Study 3 (lexical cue ↔ trait correlations) → `linguistic_homogenization_social_impact/datasets/corr_analysis.ipynb` (Tables F10–F15) and `analysis_of_the_variance_between_lexical_cues.ipynb` (ED Fig 6 + Table C7 markers).

Because notebooks are committed with their rendered outputs, you can **inspect every result without re-running** — re-running is only required if you want to extend the analysis or substitute alternative data.

---

## Questions

For any questions about the code or about how to obtain the datasets, please contact the corresponding author (see the published paper).
