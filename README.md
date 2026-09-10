# Credit Card Users Churn Prediction

An end-to-end machine learning analysis for identifying bank customers who are likely to leave their credit card service. The project combines exploratory data analysis, preprocessing, class-imbalance techniques, model comparison, and feature-importance analysis to support customer-retention decisions.

## Files

| File | Description |
| --- | --- |
| `BankChurners.csv` | Customer demographics, account information, and transaction behavior. |
| `creditcard_churn_prediction.ipynb` | The complete analysis, from data loading through business recommendations. |

## Objective

The prediction target is `Attrition_Flag`:

- `Existing Customer` - the customer remains with the bank.
- `Attrited Customer` - the customer has left the bank.

The notebook evaluates which customer characteristics and behaviors are associated with attrition, with particular attention to identifying churners rather than optimizing accuracy alone.

## What the Notebook Covers

- Dataset validation and descriptive analysis
- Distributions, relationships, and churn-focused visualizations
- Missing-value and outlier review
- Numerical scaling and categorical encoding
- Decision tree, bagging, random forest, AdaBoost, gradient boosting, and XGBoost classifiers
- Class balancing with SMOTE and random undersampling
- Hyperparameter tuning and cross-validation
- Evaluation with precision, recall, F1 score, ROC AUC, accuracy, and confusion matrices
- Feature importance and practical retention recommendations

## Installation

Python 3.9 or newer is recommended. Install the notebook dependencies with:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

## Running the Analysis

From the project directory, launch Jupyter Notebook:

```bash
jupyter notebook creditcard_churn_prediction.ipynb
```

Run the cells from top to bottom. The notebook loads `BankChurners.csv` with a relative path, so the CSV must remain in the same directory or the data-loading cell must be updated.

## Data Notes

The dataset contains demographic, relationship, credit, activity, and transaction features. `CLIENTNUM` is a unique customer identifier and should not be used as a predictive feature. Because failing to identify a likely churner can be more costly than contacting a retained customer, model selection should weigh recall and other class-sensitive metrics alongside accuracy.

## Reproducibility

The notebook is intended to be run in sequence in a fresh Python environment. Some visualizations and tuned models may take longer to execute than the exploratory cells, depending on the available CPU and memory.