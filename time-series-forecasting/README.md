# Road Accident Risk Prediction — Time Series & Machine Learning

## Project Overview
This project develops a predictive algorithm for accident risk levels and demonstrates its integration into Google Maps via a Streamlit application.  
The motivation is to **reduce traffic accidents** by forecasting road risk levels in real time, providing drivers with actionable insights and enabling authorities to implement data-driven safety strategies.

- **Domain:** Road Safety, Predictive Analytics, Legal/Ethical AI  
- **Techniques:** Data preprocessing, Neural Networks, Random Forest, Gradient Boosting, Decision Trees  
- **Deployment:** Streamlit interactive app with geospatial visualization  
- **Compliance:** Designed under the EU AI Act as a **High-Risk AI System** with legal and ethical safeguards  

---

## Objectives
1. Develop a machine learning model to predict road risk levels (low / medium / high).  
2. Integrate predictions into a Google Maps extension.  
3. Provide real-time, transparent accident risk insights.  
4. Ensure full compliance with GDPR and EU AI Act requirements for high-risk AI systems.  

---

---

## Data
Two main sources were used:
- **Breda Accident Data (2017–2023)** → accident severity, road type, weather, light conditions.  
- **ANWB Traffic Incidents Data** → traffic features, incident durations, longitude/latitude.  

These datasets enabled spatial, temporal, and environmental modeling of accident likelihood.

---

## Methodology

### Data Preprocessing
- Handled missing values and outliers.  
- Encoded categorical features (label encoding / one-hot encoding).  
- Standardized and normalized numerical features.  
- Balanced the dataset with oversampling.  

### Modeling
- **Neural Networks (Keras)** → captured complex nonlinear patterns.  
- **Decision Trees** → interpretable baseline.  
- **Random Forest** → most robust, high accuracy after hyperparameter tuning.  
- **Gradient Boosting** → boosted precision on edge cases.  

Evaluation: accuracy, precision, recall, F1-score, AUC, learning curves.  

### Deployment
- Built Streamlit app with dropdown inputs for conditions (road type, light, speed, weather, etc.).  
- Integrated geospatial visualization with Folium maps.  
- Packaged via Poetry for reproducibility.  

---

## Legal & Ethical Considerations

### Risk Classification
- Identified as **High Risk AI** under the EU AI Act.  
- Safety-critical system influencing driver decisions.  
- Handles sensitive personal and environmental data.  

### Essential Legal Requirements
- **Data Accuracy & Integrity:** keep training data current and representative.  
- **Documentation & Record-Keeping:** maintain system/process logs.  
- **Transparency & Information:** clear user guidelines, prediction limitations.  

### Compliance Checklist
- GDPR-aligned data acquisition and protection.  
- Continuous monitoring, auditing, and bias mitigation.  
- Ethical impact assessments and clear liability policies.  

---

## Results

### Model Performance (per iteration)
- **Iteration 1 – Decision Tree:** baseline accuracy ~65%, easily overfit.  
- **Iteration 2 – Random Forest (default):** ~75% accuracy, balanced recall across classes.  
- **Iteration 3 – Random Forest (tuned):** ~82% accuracy, F1 ~0.80, most robust.  
- **Iteration 4 – Gradient Boosting:** improved recall on medium-risk cases, F1 ~0.83.  
- **Iteration 5 – Neural Network (Keras):** similar accuracy (~82–83%), but required more tuning.  

**Key outcome:** Random Forest emerged as the **top-performing and most interpretable model**, chosen for deployment.

### Application Demonstration
- **Streamlit app**: takes user input (road type, weather, time of day, etc.) and outputs risk level.  
- **Google Maps UI mockups**: prediction overlay integrated into route visualization.  
- **Prototype tested** via final project presentation and demos.  

---

## Roadmap & Iteration
- **Week 4:** Data preprocessing and cleaning.  
- **Week 5–6:** Model training and evaluation.  
- **Week 7:** Deployment and final validation.  
- Beyond: Continuous monitoring, evaluation, and iteration.  

---

## Skills Demonstrated
- **Data Science:** preprocessing, feature engineering, model training.  
- **Machine Learning:** ensemble methods, neural networks, evaluation metrics.  
- **MLOps & Deployment:** Streamlit, Folium, Poetry, reproducibility.  
- **AI Governance:** compliance with EU AI Act and GDPR.  
- **Collaboration & Communication:** proposal writing, roadmap planning, final presentation, stakeholder reporting.  

---

## Next Steps
- Expand to nationwide datasets for broader scalability.  
- Integrate real-time traffic and weather APIs.  
- Explore time-series models (LSTMs, Transformers) for dynamic accident risk prediction.  
- Strengthen fairness testing and bias mitigation.  

---

## Takeaway
This project demonstrates the full AI development lifecycle:  
- Defining the problem → proposing a predictive solution.  
- Building, evaluating, and iterating ML models.  
- Deploying a user-facing application.  
- Ensuring compliance with legal and ethical AI standards.  

