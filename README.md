# Risk Classification & Machine Learning Pipeline

An end-to-end data preprocessing, feature engineering, and binary classification pipeline designed to handle high-dimensional, highly imbalanced data. Built for robust risk prediction (`gb`), missing data imputation, and reliable model evaluation.

---

## 📌 Project Overview
In financial modeling and risk assessment, datasets are frequently high-dimensional (often exceeding 550 features), plagued by missing values, and exhibit severe target class imbalance (`gb`). 
This project ingests raw data (`train_df.csv`), handles complex preprocessing steps including iterative imputation and multi-collinearity checks, and trains optimized machine learning classifiers capable of minimizing false negatives on minority classes.

---

## 🛠️ Tech Stack & Skills Demonstrated
- **Languages:** Python (Pandas, NumPy)[cite: 1]
- **Statistical Analysis & EDA:** Statsmodels (VIF calculation), Missingno, Correlation & Mutual Information analysis[cite: 1]
- **Data Preprocessing & Encoding:** Scikit-learn (`IterativeImputer`), Category Encoders (`TargetEncoder`), OneHotEncoder[cite: 1]
- **Machine Learning Models:** Scikit-learn (`RandomForestClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `VotingClassifier`), XGBoost (`XGBClassifier`)[cite: 1]
- **Model Evaluation:** ROC-AUC, Precision-Recall, F1-Score, Classification Report, Confusion Matrix[cite: 1]

---

## 🏗️ Architecture & Engineering Decisions

### 1. Exploratory Data Analysis & Imbalance Management
- **Target Distribution Analysis:** Identified extreme class imbalance in the target variable (`gb`), shifting evaluation strategy away from misleading Accuracy toward ROC-AUC and Precision-Recall metrics[cite: 1].
- **Missing Data Strategy:** Addressed extensive missing values using advanced multivariate imputation techniques (`IterativeImputer`) to preserve data distribution without dropping sparse rows or columns.

### 2. Feature Selection & Multi-collinearity
- **Variance Inflation Factor (VIF):** Utilized statsmodels to detect and eliminate multi-collinearity among high-dimensional features.
- **Feature Importance:** Evaluated feature relevance using mutual information and tree-based feature importances to reduce noise and speed up training.

### 3. Categorical Encoding & Preconditioning
- **Advanced Encoding:** Implemented `TargetEncoder` for high-cardinality categorical variables alongside standard `OneHotEncoder` to optimize representation for tree-based models and linear classifiers.

### 4. Modeling & Ensemble Architecture
- **Baseline to Ensembles:** Trained and evaluated baseline models (Logistic Regression, Decision Trees) before scaling up to robust non-linear models (`XGBoost`) and voting ensembles (`VotingClassifier`)[cite: 1].
- **Hyperparameter Tuning:** Optimized model performance and regularization parameters using `GridSearchCV` to prevent overfitting on the high-dimensional feature space[cite: 1].

---

## 📂 Repository Structure
```text
├── data/                  # Raw and processed datasets (e.g., train_df.csv)
├── notebooks/             # Jupyter Notebook containing full EDA, pipeline steps, and modeling
├── models/                # Serialized model artifacts (.joblib)
├── requirements.txt       # Project dependencies and environment specs
└── README.md              # Project documentation