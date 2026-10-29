import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📊", layout="wide")
st.title("AI-Driven Employee Attrition Analysis")
st.write(
    "Explore employee data, evaluate a machine-learning model, and estimate attrition risk. "
    "This demo supports CSV uploads and includes a small sample dataset."
)

TARGET = "Attrition"

@st.cache_data
def load_sample():
    return pd.read_csv("data/sample_employee_attrition.csv")

def normalize_target(series):
    values = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "1": 1, "true": 1, "left": 1,
        "no": 0, "n": 0, "0": 0, "false": 0, "stayed": 0
    }
    mapped = values.map(mapping)
    if mapped.isna().any():
        return None
    return mapped.astype(int)

def build_pipeline(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload employee CSV", type=["csv"])
    st.caption("Your CSV must contain a target column named Attrition (Yes/No or 1/0).")
    st.divider()
    st.write("**Demo note:** This project is for learning and decision support. Do not use predictions as the sole basis for employment decisions.")

try:
    df = pd.read_csv(uploaded) if uploaded is not None else load_sample()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if TARGET not in df.columns:
    st.error(f"Required target column '{TARGET}' was not found. Rename your target column to Attrition.")
    st.stop()

if len(df) < 10:
    st.warning("Use at least 10 rows for a meaningful demonstration.")
    st.stop()

y = normalize_target(df[TARGET])
if y is None:
    st.error("Attrition values must be Yes/No, Y/N, 1/0, True/False, Left/Stayed.")
    st.stop()

X = df.drop(columns=[TARGET]).copy()
# Drop obvious identifier fields when present; identifiers are not useful predictors.
id_cols = [c for c in X.columns if c.strip().lower() in {"employeeid", "employee_id", "id"}]
X = X.drop(columns=id_cols, errors="ignore")

if X.shape[1] == 0:
    st.error("The dataset needs at least one feature column besides Attrition.")
    st.stop()

class_counts = y.value_counts()
if len(class_counts) < 2:
    st.error("The Attrition column must contain both classes (employees who left and stayed).")
    st.stop()

st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Employee records", f"{len(df):,}")
c2.metric("Input features", f"{X.shape[1]:,}")
c3.metric("Attrition rate", f"{y.mean() * 100:.1f}%")
with st.expander("Preview data"):
    st.dataframe(df.head(20), use_container_width=True)

left, right = st.columns(2)
with left:
    st.markdown("### Attrition distribution")
    counts = y.map({0: "Stayed", 1: "Left"}).value_counts()
    fig, ax = plt.subplots()
    counts.reindex(["Stayed", "Left"]).dropna().plot(kind="bar", ax=ax)
    ax.set_xlabel("Employee outcome")
    ax.set_ylabel("Number of employees")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("### Missing values")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values found in this dataset.")
    else:
        st.dataframe(missing.rename("Missing values").to_frame(), use_container_width=True)

st.subheader("Train and evaluate the model")
test_size = st.slider("Test set size (%)", min_value=20, max_value=40, value=30, step=5)
stratify_y = y if y.value_counts().min() >= 2 else None
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42, stratify=stratify_y
    )
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.3f}")
    m2.metric("Precision", f"{precision_score(y_test, predictions, zero_division=0):.3f}")
    m3.metric("Recall", f"{recall_score(y_test, predictions, zero_division=0):.3f}")
    m4.metric("F1-score", f"{f1_score(y_test, predictions, zero_division=0):.3f}")

    try:
        auc = roc_auc_score(y_test, probabilities)
        st.caption(f"ROC-AUC: {auc:.3f}")
    except ValueError:
        st.caption("ROC-AUC is unavailable because the test split contains only one class.")

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Classification report"):
        report = classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["Stayed", "Left"], zero_division=0, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.subheader("Predict attrition for a new employee")
    st.write("Enter values below. Leave fields unchanged to use a typical value from the training data.")
    form_values = {}
    with st.form("prediction_form"):
        cols = st.columns(3)
        for idx, col_name in enumerate(X.columns):
            series = X[col_name]
            with cols[idx % 3]:
                if pd.api.types.is_numeric_dtype(series):
                    default = float(series.median()) if series.notna().any() else 0.0
                    form_values[col_name] = st.number_input(
                        col_name, value=default, key=f"num_{col_name}"
                    )
                else:
                    choices = series.dropna().astype(str).value_counts().index.tolist()
                    if not choices:
                        choices = ["Unknown"]
                    form_values[col_name] = st.selectbox(
                        col_name, choices, key=f"cat_{col_name}"
                    )
        submitted = st.form_submit_button("Predict attrition")

    if submitted:
        new_employee = pd.DataFrame([form_values], columns=X.columns)
        risk = float(pipeline.predict_proba(new_employee)[0, 1])
        predicted = int(risk >= 0.5)
        if predicted:
            st.warning(f"Model prediction: **Higher attrition likelihood** (estimated probability: {risk:.1%}).")
        else:
            st.success(f"Model prediction: **Lower attrition likelihood** (estimated probability of leaving: {risk:.1%}).")
        st.caption("This is a statistical estimate, not a certainty or a judgment about an individual employee.")

except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.info("Check that the CSV contains enough rows, both Attrition classes, and valid feature columns.")
