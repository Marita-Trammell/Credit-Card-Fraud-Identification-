#  Credit-Card Fraud Detection – Supervised Learning Capstone  
**DTSA 5509 • Master of Science in Data Science**

## 1. Problem Statement  
Credit-card fraud causes multi-billion-dollar annual losses.  
Our goal is to classify transactions as **fraudulent (1)** or **legitimate (0)** while maximising recall and keeping the false-alarm volume practical for a bank’s fraud-operations team.

## 2. Dataset  
**Source:** Dal Pozzolo A., Caelen O., Johnson R., & Bontempi G. (2015). *Credit Card Fraud Detection* (Version 1) [Data set]. Université Libre de Bruxelles — Machine Learning Group.  
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

- **Rows / Frauds:** 284 807 transactions, 492 frauds (0.172 %)  
- **Features (31):** Time, Amount, 28 PCA components V1–V28, target Class  
- **License:** Open for academic / non-commercial use (see Kaggle page)

## 3. Methodology Summary

### Data Cleaning  
- Drop 1 081 duplicates  
- Cap & z-scale Amount at €2 500 (99.9ᵗʰ percentile)  
- Engineer `Hour = (Time // 3600) % 24`  
- Zero missing values; PCA components are orthogonal → no multicollinearity  

### Exploratory Analysis  
- Fraud skews to early-morning, low-amount transactions  
- Top correlated PCs: V17 / V14 / V12 (|r| > 0.26)  
- Mann–Whitney U: *p* < 1 × 10⁻¹⁵ → distributions differ significantly  
- t-SNE shows tight fraud cluster → motivates non-linear models  

### Models & Tuning  
| Model            | Key Settings                                        | Imbalance Handling |
|------------------|-----------------------------------------------------|---------------------|
| Logistic Regression | L2, `class_weight=balanced`                      | Weights             |
| LogReg + SMOTE   | 10% minority oversample inside CV                   | SMOTE               |
| Random Forest    | `RandomizedSearchCV` (8 combos)                     | Weights             |
| XGBoost          | Grid search (`max_depth`, `subsample`, `colsample_bytree`) | `scale_pos_weight` |
| Voting Ensemble  | Soft vote of Logit + RF + XGB                       | Blend               |

**Evaluation:** 20% stratified hold-out  
**Primary metric:** Average Precision (AP); also ROC-AUC, Recall, Precision, F1

## 4. Results 
| Model           | AP    | ROC-AUC | Recall | Precision | F1     |
|----------------|-------|---------|--------|-----------|--------|
| Logistic       | 0.722 | 0.972   | 0.908  | 0.056     | 0.105  |
| + SMOTE        | 0.750 | 0.967   | 0.888  | 0.388     | 0.540  |
| Random Forest  | 0.837 | 0.966   | 0.796  | 0.839     | 0.817  |
| XGBoost        | 0.886 | 0.980   | 0.837  | 0.891     | 0.863  |
| Ensemble       | 0.855 | 0.972   | 0.867  | 0.780     | 0.821  |

**Take-away:** Tuned XGBoost adds +16 AP points vs. baseline; ensemble offers highest recall with tolerable false positives.

## 5. Discussion & Next Steps  
**Learning:** Weighted boosting beats global oversampling; engineered temporal feature adds signal.  
**Limitations:** Dataset covers only two days; PCA anonymisation hinders feature semantics.  

**Future Work:**  
- Cost-sensitive threshold tuning (dollar-loss minimisation)  
- Online/streaming boosting for real-time detection  
- Graph-based merchant-customer network features
