# Risk Classification & Machine Learning Pipeline

A pipeline for data preprocessing, feature engineering, and binary classification for risk prediction (target variable `gb`). The project covers the full workflow: exploratory analysis, imputation of missing values, handling of severe class imbalance, multicollinearity control, and model evaluation.

The source data is high-dimensional (550+ features), contains a large share of missing values, and has a strongly imbalanced target. The pipeline takes the raw dataset (`train_df.csv`), prepares it step by step, and trains classifiers aimed at minimizing false negatives on the minority class.

---

## Project Architecture

The project has a modular structure:

* **`data/`** — raw and processed datasets.
* **`notebooks/`** — a Jupyter Notebook with the complete EDA, pipeline steps, and modeling.
* **`models/`** — serialized model artifacts (`.joblib`).
* **`requirements.txt`** — project dependencies and environment specification.
  
---

## Pipeline Stages

### 1. Exploratory Data Analysis and Class Imbalance
* **Target analysis:** investigation of the `gb` distribution and identification of the class imbalance. Because of it, evaluation focuses on ROC-AUC and Precision-Recall instead of Accuracy.
* **Missing data:** multivariate imputation with `IterativeImputer`, which preserves the data distribution without dropping sparse rows or columns.

### 2. Feature Engineering and Multicollinearity
* **Multicollinearity:** detection and removal of collinear features using the Variance Inflation Factor (VIF) from `statsmodels`.
* **Feature selection:** feature importance evaluated with mutual information and tree-based metrics to reduce noise and speed up training.
* **Categorical encoding:** `TargetEncoder` for high-cardinality variables and `OneHotEncoder` for the rest.

### 3. Modeling and Evaluation
* **Baseline models:** Logistic Regression and Decision Tree.
* **Ensemble methods:** Random Forest, XGBoost, and `VotingClassifier`.
* **Hyperparameter tuning:** `GridSearchCV` with regularization to prevent overfitting in the high-dimensional feature space.
* **Metrics:** ROC-AUC, Precision-Recall, F1-Score, Classification Report, and Confusion Matrix.

---

## Usage

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the raw dataset `train_df.csv` in the `data/` folder.
3. Open the notebook from `notebooks/` and run the cells in order. Trained models are saved to `models/`.
