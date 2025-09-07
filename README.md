# BigOAI-Final-Course-1-Insurance-Premium-Prediction

This project was developed for the **[Big OAI Final Course 1 Kaggle competition](https://www.kaggle.com/competitions/big-oai-final-course-1/leaderboard?tab=public)**.  
The challenge: **predict insurance premium amounts** for policyholders given demographic, financial, and policy-related data.  

Performance was evaluated using **Root Mean Squared Logarithmic Error (RMSLE)**.  

---

## 📌 Problem Overview
- **Type:** Supervised Machine Learning – Regression  
- **Target:** `Premium Amount` (continuous variable)  
- **Metric:** RMSLE  
- **Dataset size:** ~630k rows, 48 features  

---

## 🔍 Approach

### 1. Exploratory Data Analysis (EDA)
- Examined feature distributions and target skewness.  
- Identified categorical vs. numerical features.  
- Found right-skewed variables (e.g., `Annual Income`, `Premium Amount`).  

### 2. Feature Engineering
- **Binary Encoding:** `Gender`, `Smoking Status`  
- **One-Hot Encoding:** `Marital Status`, `Education Level`, `Occupation`, `Location`, `Policy Type`, `Customer Feedback`, `Exercise Frequency`, `Property Type`  
- **Date Features:** from `Policy Start Date` → created `policy_days_since`, `policy_month`, `policy_quarter`.  
- **Log Transformations:** Applied to skewed features and target variable.  
- **Outlier Handling & Scaling:** Reduced variance and improved stability.  

### 3. Modeling
- Baseline: **LightGBM Regressor**  
- Target log-transformed during training → back-transformed with `np.expm1()` at prediction.  
- **Hyperparameter tuning** with GridSearchCV and Optuna:  
  - Tree complexity (`num_leaves`, `max_depth`)  
  - Learning rate & estimators (`learning_rate`, `n_estimators`)  
  - Regularization (`lambda_l1`, `lambda_l2`, `min_child_samples`)  
  - Subsampling (`feature_fraction`, `bagging_fraction`)  

---

## 📈 Results
- **Baseline RMSLE:** ~1.10  
- Feature engineering (especially date features & log-scaling) significantly improved model accuracy.  
- Optuna hyperparameter tuning improved stability and generalization.  

---

## 🚀 Future Improvements
- Out-of-Fold (OOF) stacking with LightGBM, CatBoost, and Ridge Regression.  
- Advanced feature interactions.
- Better handling of extreme premiums in long-tail distribution.  

---

## 🛠 Tech Stack
- Python (Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib)  
- LightGBM  
- Optuna & GridSearchCV for hyperparameter optimization  
- Jupyter Notebook  

