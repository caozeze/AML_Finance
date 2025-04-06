# Feature Importance Consistency Analysis

## Overview
This analysis compares feature importance patterns across 6 different credit scoring models:
Logistic Regression, XGBoost, MLP, Transformer, TabTransformer, FTTransformer

## Key Findings

### Most Important Features
The following features appear in the top 10 for most models:
- **interest_rate**: Appears in top 10 for 6 out of 6 models
- **credit_mix**: Appears in top 10 for 6 out of 6 models
- **outstanding_debt**: Appears in top 10 for 6 out of 6 models
- **delay_from_due_date**: Appears in top 10 for 6 out of 6 models
- **month**: Appears in top 10 for 5 out of 6 models

### Model Similarity
Based on Jaccard similarity of top features:
- **Most similar models**: Logistic Regression and FTTransformer (Jaccard=0.82)
- **Least similar models**: XGBoost and MLP (Jaccard=0.43)

### Financial Domain Consistency
- **Most consistent domain**: Spending Pattern (σ=0.425)
