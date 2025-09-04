# Data Analysis: NAC Breda — Player Evaluation


## Overview

I analyzed NAC Breda’s scouting dataset to understand **goal‐scoring potential** and turn raw stats into **recruitment insights**. This folder contains my end-to-end notebooks (preprocessing → targeted fixes → analysis) and a concise, executive-style report with the key takeaways.

## Objective

Answer a practical question with evidence: **Which player traits and on-ball behaviours best signal goal contribution, and how can I shortlist promising profiles for scouting & development?**

## What’s inside

* **`DataLab_I_Week3_NAC_Preprocessing_1.ipynb`** — ingest, schema/type audit, missing-value plan, initial cleaning.
* **`missing_value_fix.ipynb`** — focused NA handling, rare-category consolidation, post-fix QA checks.
* **`DataLab_II_Week3_NAC_Analysis.ipynb`** — EDA (distributions, correlations, position/age slices), comparisons of expected vs. actual output, and baseline modeling.
* **`data-driven_strategies.pdf`** — *Data-Driven Strategies for Goal-Scoring Success*: my final, skimmable report with figures and recommendations.
* *(optional, if present)* **`Template-Final-Report.pdf`** — alternate formatted report of the same study.

> **Skim tip:** open **`data-driven_strategies.pdf`** first for the narrative; then jump to the last sections of the **Analysis** notebook for results.

## What I did (methodology)

1. **Preprocessing**

   * Audited column types, duplicates, and missingness; documented an imputation strategy.
   * Imputed categoricals and numerics with transparent, reproducible rules.
   * Standardized/encoded features as needed and produced a clean, model-ready table.
2. **Targeted fixes**

   * Resolved edge-case columns and rare categories discovered after the first pass.
   * Re-ran data-quality checks to ensure no remaining NA/Inf and consistent ranges.
3. **Exploratory analysis**

   * Distributions and correlations of key performance metrics.
   * Slices by **position** and **age groups** (e.g., goals/90, assists/90, duels/aerials, shots-on-target %, pass accuracy).
   * **Expected vs. actual** output comparisons to flag over-/under-performance profiles.
4. **Baselines & sanity checks**

   * Simple, interpretable baselines (linear/logistic and tree-based starts) to quantify relationships and set realistic expectations.
   * Emphasis on clarity and decision support over leaderboard chasing.

## Questions I answered

* What does the **age** and **position** distribution look like?
* Which features relate most to **goals** (e.g., **shots/90**, **shots-on-target %**, **assists/90**, **duels**/aerial involvement)?
* How do **goals/90** and **assists/90** vary **by position**?
* Where do we see **over-/under-performance** vs. expectations (xG/xA when available)?
* Which patterns are strong enough to drive **shortlisting** and training focus?

## How to run locally

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab

jupyter lab
# Run notebooks in this order:
# 1) DataLab_I_Week3_NAC_Preprocessing_1.ipynb
# 2) missing_value_fix.ipynb
# 3) DataLab_II_Week3_NAC_Analysis.ipynb
```

**Data note:** If the original scouting data isn’t tracked in the repo, set the file path in the first cells of the preprocessing notebook (or point to a small sample CSV with the same schema).

## Ethics & privacy

I keep the analysis focused on performance features, avoid publishing personal identifiers, and follow good data-handling practices aligned with GDPR expectations.

## Next steps

* Add calibrated, explainable models (e.g., tree-based + SHAP) for shortlisting.
* Package feature engineering into a reusable `sklearn` Pipeline for batch scoring or deployment.
* Refresh with new seasons and track deltas year-over-year.
