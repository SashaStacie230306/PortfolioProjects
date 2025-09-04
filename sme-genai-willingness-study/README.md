# SME GenAI Willingness — Mixed-Methods Study 

## Introduction

I designed and executed a mixed-methods study to answer a practical question for SMEs: **what actually drives employees’ willingness to learn and adopt Generative AI—and how should leaders design training so adoption sticks?**
This was a team project; the materials here reflect **my individual contribution**.

---

## My Role & Contributions

* **Analysis owner.** Built the primary notebook (`analysis.ipynb`): data cleaning, recoding, descriptives, **correlations, t-tests, one-way ANOVA, and a simple regression**; produced all figures.
* **Evidence → policy.** Co-authored the **policy paper**, translating results into concrete training and rollout steps for SME leadership.
* **Study coordination.** Helped define the week-by-week plan, tracked progress (see `Roadmap_Project.png`), and consolidated outputs for this portfolio cut.
* **Individual sub-study.** Led an analysis on **education level vs. willingness** to test a common assumption; wrote up findings as a short paper.

---

## Key Findings (Plain Language)

* **Relevance and managerial support are decisive.** Willingness rises when people see **clear, role-specific use cases** and when managers **explicitly endorse safe experimentation**.
* **Degrees don’t gate adoption.** Education level **was not a statistically significant predictor** of willingness in my analysis once relevance and support were considered.
* **Barriers are practical.** Time constraints, uncertainty about approved tools, and privacy/compliance concerns reduce willingness.
* **Preferred training format.** Short, hands-on modules using **approved tools**, with reusable templates and immediate application to the job.

**Implication for leaders:** shift away from degree filters; design **role-tied, manager-backed, governance-aware** learning paths and measure **usefulness and usage**, not just attendance.

---

## Methods

* **Design:** mixed methods—online **survey** (Likert scales + demographics + open text) and a set of **semi-structured interviews**.
* **Quant analysis:** cleaning, descriptives, **correlation**, **t-tests**, **one-way ANOVA**, **simple regression** on a composite **Willingness Score** (interest + time willing to invest). Effect sizes reported; results visualized.
* **Qual analysis:** rapid coding of open responses/interviews to surface perceived benefits, barriers, and training preferences.
* **Limits (transparent):** cross-sectional, self-report data; subgroup sizes vary; findings show associations (not causation).

---

## Deliverables

```
sme-genai-willingness/
├─ analysis.ipynb                     # my notebook: cleaning → stats → visuals
├─ final_research_paper.pdf           # full narrative: methods, results, discussion
├─ Policy_paper_DSAI_Group1.pdf       # actionable recommendations for SME leaders
├─ Individual_Reserach_SS_V1 (2).pdf  # education-level sub-study (my write-up)
└─ Roadmap_Project.png                # 8-week plan we executed
```

---

## Representative Evidence

* **Education vs. willingness:** One-way ANOVA showed **no significant differences** across HS/Bachelor/Master groups (α=0.05). Practical reading: don’t gate training by degree.
* **Motivation linkage:** “Interest” and “time willing to commit” are **positively correlated** (moderate strength), but not sufficient on their own—organizational support matters.
* **Qualitative echo:** Interviews emphasize **augmentation, not replacement**, and call for **clear governance** (privacy, IP, review) to reduce hesitation.

*(Full tables, plots, and effect sizes are in `analysis.ipynb` and the papers.)*

---

## Recommendations I Stand Behind

1. **Make it concrete:** map **2–3 GenAI use cases per role/team**; provide prompt playbooks and worked examples.
2. **Signal safety:** visible manager sponsorship and simple, human-readable rules for **privacy, IP, and human-in-the-loop review**.
3. **Lower the cost to try:** bite-sized sessions, office hours, and internal champions who unblock colleagues quickly.
4. **Train for quality:** cover prompt hygiene, evaluation checklists, and when to escalate to expert review.
5. **Measure what matters:** track **usage and perceived usefulness** (and examples of work saved/improved), not just course completion.

---

## How to Run the Notebook

```bash
# Python 3.10+
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -U pip
pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab

jupyter lab
# open and run: analysis.ipynb
```

**Data & ethics:** raw survey/interview data isn’t committed. The notebook expects **anonymized/aggregated CSVs** or a local export with the same variable names—adjust file paths in the first cell. We followed consent and GDPR-aligned handling; see the papers for details.

---

## What This Project Demonstrates (for Recruiters)

* **Problem framing → evidence → action.** I define a real adoption question, select fit-for-purpose methods, and land on clear, defensible recommendations.
* **Technical fluency with restraint.** I run appropriate statistics and visualize them cleanly—no over-claiming, no unnecessary complexity.
* **Translation to policy.** I turn findings into concrete steps leaders can use next quarter (role mapping, templates, governance, measurement).
* **Professional packaging.** Clean repo structure, runnable notebook, and polished PDFs—easy to skim, easy to reproduce.

---

## One-Line Takeaway

**Don’t gate GenAI training by degree.** Make it **role-relevant**, **manager-backed**, and **safe by design**—and people will want to learn.
