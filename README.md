# Credit Scoring with Imbalanced Data: A Comparison of Traditional and Deep Learning Models

This repository contains the full codebase and resources for the COMP0162 project submitted as part of the MSc Data Science and Machine Learning programme at UCL.

## 📁 Project Structure

```
.
├── data/                          # Raw and preprocessed dataset files (not included)
├── models/                        # Saved model weights
├── shap/                          # SHAP analysis outputs
├── resources/                     # Helper functions or external scripts
├── 1_eda.ipynb                                    # Exploratory Data Analysis
├── 2_over_sampling.ipynb                          # Random Oversampling & SMOTE
├── 3_standardisation_and_multicollinearity_check.ipynb   # Z-score + VIF
├── 4_baseline_model_training.ipynb               # LR & XGBoost training
├── 5_train_mlp_transformer.ipynb                 # MLP & Transformer training
├── 6_statistical_test.ipynb                      # Wilcoxon significance tests
├── 7_explanability.ipynb                          # SHAP feature attribution
├── .gitignore
└── best_impute_model_weights.pth                 # Model checkpoint (if applicable)
```

## 🚀 How to Run

### Step 1: Environment Setup

We recommend using **Python 3.12** with virtual environments.

### Step 2: Run Notebooks in Order

Ensure dataset files are placed under `./data/` folder. Then execute notebooks in this order:

1. `1_eda.ipynb`
2. `2_over_sampling.ipynb`
3. `3_standardisation_and_multicollinearity_check.ipynb`
4. `4_baseline_model_training.ipynb`
5. `5_train_mlp_transformer.ipynb`
6. `6_statistical_test.ipynb`
7. `7_explanability.ipynb`

Each notebook is self-contained and may save intermediate results to local folders.

## 🧠 Models Implemented

- Logistic Regression (LR)
- XGBoost
- Multilayer Perceptron (MLP)
- Transformer
- TabTransformer
- FT-Transformer

These models are tested under both:
- Original imbalanced data
- Resampled data using SMOTE and Random Oversampling (ROS)

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- AUC (ROC)
- Wilcoxon Rank-Sum Test (Statistical Significance)
- SHAP-based Feature Importance Analysis

## 📈 Explainability

SHAP (SHapley Additive Explanations) is used to interpret model outputs.  
Top-10 important features per model are compared for consistency and domain relevance.

Related notebook:
```text
7_explanability.ipynb
```

## 📎 Project Report

The full report is submitted as a separate file:
```
credit_score_report_COMP0162.pdf
```

It includes background, methodology, results, and discussion.

## ⚖️ Ethical Considerations

- No personal, sensitive, or identifiable data was used.
- Publicly available dataset only.
- The completed ethical form is included as:
```
Ethical Risk Identification Form.docx
```

---

© 2025 Cao Ze | MSc DSML | University College London  
*For academic use only*
