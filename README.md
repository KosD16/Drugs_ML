# Drugs_ML
This repository contains the implementation of a machine learning study focused on predicting polysubstance use patterns (phenotypes) rather than isolated drug use. The analysis utilizes the UCI Drug Consumption (Quantified) dataset.

Project Overview

Traditional drug use models often treat each substance as an independent target, failing to capture the clinical reality of co-use. This project introduces a behavioral phenotype discovery approach:

Clustering: Used Jaccard Distance and Balanced K-medoids to group 18 substances into 4 distinct co-use phenotypes based on actual user behavior.

Target Definition: A user is identified as "active" in a phenotype if they consumed at least two drugs from that group within the last year.

Leakage Control: Implemented a Sequestered Design, isolating 30% of the data for cluster generation and 70% for model training/evaluation.

Tech Stack & Methodology
Language: Python

Libraries: Scikit-learn, XGBoost, Pandas, NumPy, SHAP (for interpretability).

Models Evaluated: Logistic Regression, Extra Trees, Random Forest, HistGradientBoosting, XGBoost, and MLP.

Handling Class Imbalance: Used Cost-Sensitive Learning (class weighting) and Stratified Cross-Validation to manage rare illicit drug usage classes.

Key Findings
Top Performer: Logistic Regression consistently outperformed or matched complex models, achieving a Balanced Accuracy of 0.75 - 0.82.

Interpretability: SHAP analysis revealed that while personality traits (Neuroticism, Conscientiousness) are significant, Demographics (Age, Country) remain the strongest predictors of co-use behavior.

The "Ceiling" Effect: Results suggest a performance ceiling for psychometric data, indicating the need for social and environmental factors in future predictive modeling.

Repository Structure
analysis.py / notebook.ipynb: Core machine learning pipeline.

data/: Dataset information (UCI source).

results/: Visualizations of SHAP values and model performance metrics.
