# Football — EPL Player Value & Match Outcome Prediction

## Overview

I worked on two applied sports-analytics tasks using English Premier League (EPL) data:

1. build a **player value ranking** for the 2014–15 season, and
2. **predict match outcomes** (Win/Draw/Loss from the home team’s view) for 2015–16.
   This project let me show end-to-end work: feature engineering, model training, evaluation, and packaging results.&#x20;

---

## What I built 

* **Custom player “value\_score”.** I combined standardized performance indicators (e.g., goals, assists) and player attributes (e.g., finishing, vision, close-range shot accuracy) into a single score and ranked the league. Outputs saved to CSV for quick review.&#x20;
* **Match classifier.** I engineered team-form and match-stat features (avg goals for/against home & away, goal differences, possession, fouls, cards, corners, crosses) and trained **Random Forest** and **XGBoost** models to predict W/D/L; draws were hardest. Best validation accuracy landed around **58%**.&#x20;

---

## Files in this folder

```
football/
├─ EPL_Project_Report (1).docx         # project write-up (methods, features, results)
├─ player_list_submission.ipynb        # notebook to compute value_score & export top/bottom lists
├─ prediction_submission.ipynb         # notebook to engineer features & train W/D/L classifier
├─ prediction_submission.csv           # model predictions (submission/export)
└─ prediction_submission_2.csv         # alternative run / tuned variant
```

> If you’re skimming: open the **DOCX** for a narrative of both parts, then peek at the last cells in each notebook to see how I export results.&#x20;

---

## Method (short & specific)

### Part 1 — Player Value Ranking (2014–15)

* **Normalize & aggregate.** I z-scored key performance stats and technical attributes to a common scale, then **summed** them to form `value_score`.&#x20;
* **Rank & export.** Sorted players by `value_score` to produce **Top 10** / **Bottom 10** lists and saved the table (e.g., `player_list_submission.csv`).&#x20;

### Part 2 — Match Outcome Prediction (2015–16)

* **Target.** Home result encoded as **1 = Win, 0 = Draw, −1 = Loss**.&#x20;
* **Features.** Team form & match stats: averages of goals scored/conceded (home/away splits), goal differences, **possession, fouls, cards, corners, crosses**, etc.&#x20;
* **Models.** **Random Forest** and **XGBoost**; basic imputation (mean/zero) to keep train/test aligned. **Wins** were easier; **draws** were most challenging. **\~58%** validation accuracy in the best run.&#x20;

---

## How to run locally

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyterlab

jupyter lab
# Open and run:
# 1) player_list_submission.ipynb      # builds value_score and exports CSV
# 2) prediction_submission.ipynb       # engineers features, trains models, exports predictions
```

**Data note:** if raw EPL data isn’t committed, point the first cells in each notebook to your local dataset paths (season 2014–15 for player ranking; 2015–16 for match prediction).

---

## Results snapshot

* **Player ranking:** reproducible `value_score` with clear Top/Bottom lists (exported to CSV for easy review).&#x20;
* **Match prediction:** best validation accuracy ≈ **58%**; wins predicted better than draws. Further gains likely with richer context (injuries, formations, rest days).&#x20;

---

## What I’d improve next

* Add **context features** (formations, injuries, travel/rest) and **team-strength priors** (Elo) to help with draws.&#x20;
* Calibrate probabilities and try **ordinal** modeling for W/D/L ordering.
* Package the feature engineering into a small `sklearn` `Pipeline` and add unit tests for the transforms.

---

## Skills on display

* Feature engineering for sports data (form splits, match stats)
* Model selection & evaluation (RF/XGBoost; class difficulty awareness)
* Metric design (`value_score`) and reproducible exports (CSV)
* Clear reporting (problem framing → methods → results)&#x20;
