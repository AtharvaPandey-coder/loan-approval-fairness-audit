# Loan Approval Model with a Fairness Audit

A PyTorch neural network (ANN) that predicts whether a person earns more than $50K (used as a stand-in for loan approval), then checks whether the model treats men and women differently and reduces that gap.

> **Status:** In progress. Currently at the data exploration and cleaning stage.

## Project goal

1. Train an ANN on tabular data and compare it with Logistic Regression and Gradient Boosting baselines.
2. Audit the model for gender bias using fairness metrics (demographic parity and equalized odds).
3. Reduce the bias and measure the accuracy vs. fairness trade-off.
4. Show the before/after results in a Streamlit dashboard.

## Dataset

- **Name:** Adult Census Income (UCI Machine Learning Repository)
- **Source:** [Kaggle: uciml/adult-census-income](https://www.kaggle.com/datasets/uciml/adult-census-income)
- **File:** `adult.csv`
- **Size:** 32,561 rows, 15 columns
- **Task:** Binary classification. Predict whether `income` is `>50K` or `<=50K`.

Important: this dataset predicts income, not real loan approval. In this project, "approved" means "income above $50K".

### Columns

| Column | Type | Description |
|---|---|---|
| age | number | Age of the person |
| workclass | text | Type of employer (Private, Self-emp, Gov, ...) |
| fnlwgt | number | Census sampling weight (dropped, not useful for prediction) |
| education | text | Highest education level |
| education.num | number | Education level as a number |
| marital.status | text | Marital status |
| occupation | text | Job type |
| relationship | text | Role in household (Husband, Wife, ...) |
| race | text | Race |
| sex | text | Male or Female (**sensitive attribute used for the fairness audit**) |
| capital.gain | number | Income from investments |
| capital.loss | number | Losses from investments |
| hours.per.week | number | Working hours per week |
| native.country | text | Country of origin |
| income | text | **Target:** `<=50K` or `>50K` |

### Known data issues
- Missing values are written as `?` in `workclass`, `occupation` and `native.country` (about 7% of rows). They are kept and filled with the category `Unknown`, so no rows are lost and the male/female balance is unchanged.
- About 24 exact duplicate rows were removed to avoid the same record appearing in both the training and test sets.
- The classes are imbalanced: far more people earn `<=50K` than `>50K`.
- Men and women are not equally represented, and their approval rates differ in the raw data. This is what the fairness audit examines.

## Project plan

- [x] Download the dataset and load it
- [ ] Explore the data (EDA)
- [ ] Clean the data (handle `?`, check the effect of dropping rows on the male/female balance)
- [ ] Encode and scale features
- [ ] Train the ANN with `Dataset` and `DataLoader`
- [ ] Train the baselines (Logistic Regression, Gradient Boosting)
- [ ] Audit fairness by gender
- [ ] Reduce bias and re-measure
- [ ] Build the Streamlit dashboard
- [ ] Write the final results and limitations

## Tech stack

Python, pandas, numpy, scikit-learn, PyTorch, fairlearn, matplotlib, seaborn, Streamlit

## Project structure

```
loan-approval-fairness-audit/
├── adult.csv          # dataset
├── project.ipynb      # exploration, training and fairness audit
├── app.py             # Streamlit dashboard
├── results.csv        # saved results (created by the notebook)
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn torch fairlearn matplotlib seaborn streamlit
```

1. Open `project.ipynb` and run all cells from top to bottom.
2. Start the dashboard with `streamlit run app.py`.

## Results

To be added after training.

## Limitations

To be added after the fairness audit.
