# Arboreal — CNN + Explainable AI for Urban Tree Identification

## Project Overview
Arboreal is a computer vision project focused on identifying common urban tree species (oak, maple, pine, palm, birch) from photographs.  
The project combines **Convolutional Neural Networks (CNNs)** with **Explainable AI (XAI)** techniques to provide both predictions and visual justifications.  

The aim was not only to build a classifier, but also to demonstrate how **interpretable AI** can build trust among users and provide value to stakeholders such as urban planners, environmental NGOs, and citizens.

This repository showcases the full end-to-end process:  
- Framing the problem through a **Creative Brief**.  
- Conducting **market research** and **stakeholder analysis**.  
- Building and iterating CNN models.  
- Implementing **Grad-CAM** for explainability.  
- Evaluating with statistical tests, A/B testing, and fairness checks.  
- Designing and testing prototypes with users.  
- Reflecting on outcomes and planning future steps.  

---

## My Role
I owned the project end-to-end, including:  
- Drafting the Creative Brief and framing the business need.  
- Conducting market research and identifying stakeholder expectations.  
- Implementing CNNs and Grad-CAM explainability in PyTorch.  
- Designing and running statistical tests (t-tests) to validate model improvements.  
- Creating A/B testing plans and prototypes to evaluate user trust.  
- Producing fairness infographics and DAPS analysis to highlight ethical issues.  
- Summarizing results in case studies and reports.  

---

## Iterative Development — Five Stages

### Iteration 1 — Baseline Feasibility
- Implemented a baseline CNN (TinyCNN) trained on heterogeneous images.  
- Accuracy: ~60–65%.  
- Grad-CAM revealed the model often focused on **background elements** (sky, pavement) instead of the leaf.  
- Outcome: Proof-of-concept worked, but showed need for better preprocessing and augmentation.  
- Prototype demo: `assets/videos/Initial_prototype.mp4`.

---

### Iteration 2 — Data Augmentation and Cropping
- Added augmentations (flips, lighting adjustments, random crops).  
- Tightened focus on leaf area to reduce background noise.  
- Accuracy improved to ~70–72%.  
- Independent samples t-tests confirmed the improvement was statistically significant (p < 0.05).  
- Grad-CAM maps showed more attention on leaf **lamina** but still distracted by shadows.  

---

### Iteration 3 — Architectural Refinement
- Expanded CNN depth slightly for stronger feature extraction.  
- Introduced pooling schedule for more robust representations.  
- Accuracy rose to ~75–77%.  
- False positives reduced, but similar-looking species (maple vs oak) remained challenging.  
- Grad-CAM showed clearer attention on **venation patterns**.  

---

### Iteration 4 — Integration of Explainability
- Implemented **Grad-CAM overlays** into the prototype UI.  
- Accuracy reached ~80–82%.  
- A/B testing showed **trust score increased by ~25%** when explanations were presented.  
- Users reported greater willingness to accept predictions when heatmaps matched leaf structures.  

---

### Iteration 5 — Final Prototype and Evaluation
- Refined preprocessing pipeline; balanced dataset to mitigate bias.  
- Integrated full XAI pipeline with overlays, case study reporting, and fairness infographic.  
- Achieved ~85% accuracy with balanced macro-F1.  
- Users in testing confirmed explanations made them more likely to adopt the system.  
- Prototype demo: `assets/videos/Final_prototype.mp4`.  

---

## Technical Highlights

### Model Architecture
- **TinyCNN baseline** in `src/model.py`:  
  - Two convolutional layers with pooling.  
  - Adaptive pooling and linear classification head.  
- Ready to extend to **MobileNet** or **EfficientNet** for improved performance.

### Explainability
- **Grad-CAM** implemented in `src/xai.py`.  
- Produces class-specific saliency maps showing which leaf regions influenced predictions.  
- Example outputs in `assets/xai/`.

### Evaluation
- Accuracy and macro-F1 tracked across all 5 iterations.  
- **Independent Samples T-tests** confirmed statistical significance of accuracy improvements.  
- **A/B Testing** validated user trust benefits of XAI.  
- **Fairness Analysis** documented dataset imbalance and mitigation.

---

## Results Summary

| Iteration | Key Change                        | Accuracy | Insights from Grad-CAM                                |
|-----------|-----------------------------------|----------|-------------------------------------------------------|
| 1         | Baseline TinyCNN                  | 60–65%   | Focused on background, not leaf                       |
| 2         | Augmentation + Cropping           | 70–72%   | More attention on lamina, but shadows distracting     |
| 3         | Deeper architecture               | 75–77%   | Venation patterns attended, fewer false positives     |
| 4         | Grad-CAM integrated in prototype  | 80–82%   | Explanations boosted trust (+25% in A/B testing)      |
| 5         | Balanced dataset + final pipeline | ~85%     | Stable accuracy, high user trust, ready for scaling   |

---

## Stakeholder Value

### Urban Planners
- Faster, cheaper tree inventories.  
- Transparent predictions for decision support.  

### Environmental NGOs
- Engagement tool for biodiversity campaigns.  
- Visual explanations enhance storytelling.  

### Citizens & Youth
- Fun, educational experience that builds biodiversity literacy.  
- Trustworthy outputs due to explanations.  

---

## Fairness and Ethics (DAPS)
- **Data**: Public datasets and citizen-science images, curated and augmented.  
- **Accountability**: Grad-CAM explanations improve transparency.  
- **Privacy**: No personal data; tree photos only.  
- **Sustainability**: Direct alignment with UN SDGs on sustainable cities and biodiversity.  

---

## Next Steps
- Expand taxonomy beyond 5 species.  
- Fine-tune pretrained CNNs (MobileNet, EfficientNet).  
- Add more XAI methods (Score-CAM, Guided Backprop).  
- Deploy mobile-ready version for field use.  
- Incorporate gamification and citizen-science features.  

---

## Key Takeaways
This project demonstrates a **complete machine learning workflow** with five iterations of improvement. It highlights skills in:  
- **Modeling**: CNNs, PyTorch training pipelines.  
- **Explainability**: Custom Grad-CAM implementation.  
- **Evaluation**: Statistical significance testing, A/B testing, fairness analysis.  
- **Design**: Creative brief, stakeholder analysis, prototype design.  
- **Communication**: Translating technical results into insights for recruiters, managers, and non-technical audiences.

The iterative journey from baseline to a final explainable prototype shows not only technical ability but also the capacity to learn, reflect, and adapt — exactly what recruiters value.
