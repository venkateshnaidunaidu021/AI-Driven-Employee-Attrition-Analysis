# AI-Driven Employee Attrition Analysis Using Machine Learning

A beginner-friendly machine-learning project that explores employee attrition data, trains a Random Forest classifier, evaluates its performance, and provides an interactive Streamlit dashboard.

## Features

- Upload an employee dataset in CSV format or use the included sample data.
- Review employee records, feature count, attrition rate, and missing values.
- Train a Random Forest classification pipeline with imputation and one-hot encoding.
- View accuracy, precision, recall, F1-score, ROC-AUC (when available), and a confusion matrix.
- Enter employee feature values in the dashboard to get an estimated attrition probability.
- Handles numeric and categorical columns and ignores common employee ID columns.

## Project structure

```text
AI-Driven-Employee-Attrition-Analysis/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── sample_employee_attrition.csv
```

## Requirements

- Python 3.10 or newer
- pip

## Run locally

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (recommended).

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install packages:
```bash
pip install -r requirements.txt
```

5. Start the dashboard:
```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually `http://localhost:8501`.

## Dataset format

The uploaded CSV must include a column named `Attrition`. Accepted target values include `Yes`/`No`, `Y`/`N`, `1`/`0`, `True`/`False`, and `Left`/`Stayed`. Each row should represent one employee and each other column should be a feature, such as Age, Department, JobRole, MonthlyIncome, JobSatisfaction, or YearsAtCompany.

The included `sample_employee_attrition.csv` is **synthetic demo data** created for this repository. It is not a real employee dataset and should not be used to draw real-world HR conclusions. For academic work, consider a properly licensed public dataset such as IBM HR Analytics Employee Attrition & Performance, and document its source and license.

## Methodology

1. Load employee records from CSV.
2. Separate the `Attrition` target from input features.
3. Split data into training and testing sets (default: 70%/30%).
4. Impute missing values and encode categorical features in a scikit-learn pipeline.
5. Train a Random Forest classifier with class weighting.
6. Evaluate predictions on the held-out test set.
7. Use the trained model to estimate attrition probability for entered employee details.

## Evaluation metrics

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** among employees predicted to leave, the proportion who actually left.
- **Recall:** among employees who left, the proportion identified by the model.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking quality across classification thresholds, when both classes are present in the test set.

Metrics depend on the dataset and train/test split. Do not claim a specific accuracy before running the model on your selected dataset.

## Important limitations and responsible use

- Predictions are estimates, not certainties.
- Synthetic demo data is only for testing the application.
- Real datasets can contain sampling bias, historical bias, missing information, and leakage.
- Do not use this model as the sole basis for hiring, firing, promotion, or other employment decisions.
- Use authorized data only, protect personal information, and review fairness and model performance before any real-world use.

## Possible future enhancements

- Compare Random Forest with Logistic Regression, Decision Tree, Gradient Boosting, and XGBoost.
- Add cross-validation and hyperparameter tuning.
- Add feature-importance and explainability visualizations.
- Evaluate fairness and performance across relevant groups with appropriate safeguards.
- Add automated tests and a CI workflow.

## GitHub upload

Create a repository named `AI-Driven-Employee-Attrition-Analysis`, then upload the files in this folder. If using Git:

```bash
git init
git add .
git commit -m "Add employee attrition analysis ML project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/AI-Driven-Employee-Attrition-Analysis.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## License

Choose a license before publishing. For example, an MIT license is suitable for many educational projects, but confirm it fits your intended use and the data license.
