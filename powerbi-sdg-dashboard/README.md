# Power BI — SDG 6 (Clean Water & Sanitation) Dashboard

## Overview

I built an interactive dashboard around **SDG 6** to answer one question:
**How do improvements in drinking-water supply and WASH practices relate to diarrheal-disease mortality (especially in infants), and how does this vary by income level?**

The report is split into a **macro view** (country/region level) and a **micro view** (household-level WASH factors from a case–control perspective). It’s designed for fast executive skim with slicers and bookmark toggles for deeper analysis.

---

## What’s on the pages

### 1) Global Access & Context (Overview)

* **Map:** *Number of people without access to safe drinking water (2020)*, choropleth by country/region.
* **KPI card:** *Median people without access* ≈ **848.13K** (for the selected filter context).
* **Slicers:** **Entity** (country/region grouping).
* **SDG tile:** SDG 6 icon and guiding question at the top.

### 2) Macroeconomic Relationships (Country/Region level)

* **Scatter + trendline:**

  * *Death rate from unsafe water vs. GDP per capita (2019)* → **negative** relationship.
  * *Diarrheal diseases death rate in children vs. GDP per capita (2019)* → **negative** relationship.
  * *Improved water sources vs. GDP per capita (2000–2020)* → **positive** relationship.
* **Correlation gauges (bookmark toggle “Show Correlation”):**

  * Deaths from unsafe water vs. GDP per capita: **−0.28**
  * Diarrheal deaths in children vs. GDP per capita: **−0.31**
  * Improved water source vs. GDP per capita: **+0.70**

> Interpretation: higher-income contexts tend to have **lower water-related death rates** and **greater access to improved water sources**.

### 3) Disease Burden by Age

* **Stacked area chart:** *Deaths from diarrheal diseases by age group* (Under-5, 5–14, 15–49, 50–69, 70+).
* **Slicers:** **Entity**, **Year** for temporal and geographic focus.
* **Legend card:** clarifies “**Cases** = infants who died from diarrheal disease” vs “**Controls** = infants who survived their first year of life”.

### 4) Micro Determinants — WASH Factors (Case–Control)

**Buttons:** *Show Graph* / *Show Correlation* (bookmarks)

* **Water Supply Factors**

  * *Household water source* (Improved vs. Unimproved) — Case/Control %
  * *Drinking-water storage* (Safe vs. Unsafe) — Case/Control %
  * *Household treatment practices* (Treat vs. Do not treat) — Case/Control %
* **Economic Indicators**

  * *Distance to access clean water* (Within 1 km vs Above 1 km) — Case/Control %
  * *Household water accessibility* (Accessible vs Not accessible) — Case/Control %
* **Sanitation & Hygiene Factors**

  * *Sanitation status* (Improved vs Unimproved) — Case/Control %
  * *Critical times for handwashing* (none / <3 / ≥3 critical times) — Case/Control %
  * *Handwashing facility near latrine* (Yes/No) — Case/Control %

> Pattern you’ll see: **improved sources, safe storage, water treatment, better sanitation, and consistent handwashing** are all associated with **lower case percentages** relative to controls.

---

## How to use

1. Open `SDG Dashboard final present (1).pbix` in Power BI Desktop.
2. Use the **Entity** and **Year** slicers to focus on a country/region and period.
3. On the macro page, toggle **Show Graph / Show Correlation** to switch between scatterplots and correlation gauges.
4. Hover over marks for tooltips; right-click visuals for **Drill** options where applicable.
5. Export screenshots to `assets/` for your portfolio if needed.

---

## Data model (conceptual)

* **Facts:** access to safe drinking water, deaths from unsafe water, diarrheal disease (including child-specific), and GDP per capita (PPP).
* **Dimensions:** Entity (country/region), Year/Date, Age Group, and WASH factors (household source, storage, treatment, sanitation, handwashing).
* Time intelligence and correlations are calculated in measures; medians are used for robust KPIs where appropriate.

---

## Design choices

* **Color & layout:** gradient background and rounded cards for a modern, readable aesthetic.
* **Bookmarks & toggles:** “Show Graph / Show Correlation” provide two lenses on the same story.
* **Gauges for correlation:** quick, at-a-glance narrative (−1 → +1).
* **Slicers kept minimal** (Entity, Year) to keep the entry experience simple.

---

## Notes & sources

* Indicators include GDP per capita (PPP), deaths from unsafe water, diarrheal mortality (incl. children), access to improved water sources, and household-level WASH variables.
* Source references are documented inside the PBIX (e.g., WHO/UNICEF/World Bank/OWID style series). Sensitive or licensed data isn’t redistributed in the repo.

---

## One-liner summary 

**Higher GDP contexts tend to have lower water-related mortality and higher access to improved water sources; at the household level, improved supply, safe storage/treatment, better sanitation, and consistent handwashing are associated with fewer infant diarrhea deaths.**
