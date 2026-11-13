# Customer Churn Prediction

Predicting which telecom customers are likely to leave, so a retention team can take action before they do.

**Result:** Tuned Gradient Boosting model with ROC-AUC ≈ 0.85 that identifies approximately 90% of churners on a held-out test set (decision threshold: 0.15, selected to prioritize recall).

![Feature importance](images/12_feature_importance.png)

## The Problem

Acquiring a new customer can cost significantly more than retaining an existing one. About 1 in 4 customers in this dataset churns, so the business needs to know **who is at risk and why**.

The model assigns a churn probability to each customer, allowing the retention team to identify high-risk customers and target them with suitable retention offers.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (public dataset): 7,043 customers and 21 columns covering demographics, services, contracts, billing details, and customer churn status.

## Approach

| Step | What I did | Notebook |
|---|---|---|
| 1. Cleaning | Fixed `TotalCharges` stored as text (11 blanks, all new customers), dropped the customer ID column, merged redundant labels, and encoded the target | `01_cleaning.ipynb` |
| 2. EDA | Explored churn patterns and identified potential churn drivers using segmented charts | `02_eda.ipynb` |
| 3. Baseline | Built a Logistic Regression model using a scikit-learn Pipeline to prevent data leakage | `03_modeling.ipynb` |
| 4. Model selection | Compared Logistic Regression, Random Forest, and Gradient Boosting using 5-fold stratified cross-validation | `03_modeling.ipynb` |
| 5. Tuning | Optimized the Gradient Boosting model using GridSearchCV | `03_modeling.ipynb` |
| 6. Threshold selection | Selected the decision threshold using cross-validated training predictions and the F2-score, then evaluated the test set once | `03_modeling.ipynb` |

## Key Findings

![Churn by contract](images/02_churn_by_contract.png)

- **Contract type is the strongest driver:** Month-to-month customers churn at approximately 43%, compared with approximately 3% for two-year contract customers.
- **The first year is the riskiest:** Churn decreases as customer tenure increases.
- **Internet and payment methods matter:** Fiber optic and electronic-check customers show higher churn rates, while customers with tech support or online security tend to churn less often.
- **Gender has almost no effect:** Churn rates are similar across genders.

*These findings represent associations, not proven causes. Features can overlap. For example, fiber optic customers may also have higher monthly charges and more month-to-month contracts.*

## Model Results

Test set: 1,409 customers.

| Model | CV ROC-AUC | Train ROC-AUC |
|---|---:|---:|
| Logistic Regression (balanced) | 0.84 | 0.85 |
| Random Forest (balanced) | 0.83 | 1.00 (overfitting) |
| Gradient Boosting (tuned) | 0.85 | 0.87 |

**Final model: Tuned Gradient Boosting (test set, decision threshold 0.15)**

| Metric | Test Set Result |
|---|---:|
| ROC-AUC | 0.85 |
| Precision | 0.44 |
| Recall | 0.90 |
| F1-score | 0.59 |

![Confusion matrix](images/11_confusion_matrix_final.png)

### Why optimize for recall?

Approximately 26.5% of customers in the dataset churn.

A model that predicts "no churn" for every customer would achieve approximately 73.5% accuracy but would fail to identify any actual churners.

For this project, I prioritized **recall** because identifying customers who are likely to leave gives the retention team an opportunity to intervene.

I selected a decision threshold of **0.15** using cross-validated training predictions and the F2-score, which gives more importance to recall than precision.

The trade-off is:

- **Recall ≈ 90%:** The model identifies most customers who actually churn.
- **Precision ≈ 44%:** Of the customers predicted to churn, approximately 44% actually churn.
- A lower threshold identifies more potential churners but also increases false positives.

This helps demonstrate the balance between identifying at-risk customers and avoiding unnecessary retention offers.

![Threshold trade-off](images/10_threshold_tradeoff.png)

## Try It

### 1. Clone the repository

```bash
git clone https://github.com/Pooja012/customer-churn-prediction.git

cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows Git Bash:**

```bash
source venv/Scripts/activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run predictions

```bash
# High-risk customer example
python predict.py

# Low-risk customer example
python predict.py --file examples/low_risk_customer.json
```

## Technologies Used

- **Python:** Main programming language
- **Pandas and NumPy:** Data cleaning and manipulation
- **Matplotlib and Seaborn:** Data visualization
- **Scikit-learn:** Machine learning, model selection and evaluation
- **GridSearchCV:** Hyperparameter tuning
- **Joblib:** Saving and loading the trained model

## Project Objective

The objective of this project is to demonstrate how machine learning can help telecom companies identify customers who are likely to leave, understand churn patterns, and make data-driven customer retention decisions.
