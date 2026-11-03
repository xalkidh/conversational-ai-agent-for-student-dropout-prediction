import pickle
import numpy as np
import pandas as pd
from langchain.tools import tool

# Φόρτωση μοντέλου και scaler
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@tool
def predict_dropout(input_data: str) -> str:
    """
    Προβλέπει αν ένας φοιτητής θα εγκαταλείψει τις σπουδές του.
    Το input πρέπει να είναι JSON string με τα εξής πεδία:
    Age, Gender, Year_of_Study, GPA, Attendance_Rate, 
    Study_Hours_per_Day, Stress_Index, Financial_Support,
    Part_Time_Job, Family_Income_Level, Internet_Access
    """
    import json

    try:
        data = json.loads(input_data)
        df = pd.DataFrame([data])

        # Feature engineering (ιδιο με το HW1)
        df["study_attendance_ratio"] = df["Study_Hours_per_Day"] * df["Attendance_Rate"] / 100
        df["stress_gpa_ratio"] = df["Stress_Index"] / (df["GPA"] + 0.1)

        # Categorical και numerical features
        cat_features = df.select_dtypes(exclude=["number"]).columns.tolist()
        num_features = df.select_dtypes(include=["number"]).columns.tolist()

        # One-hot encoding
        df = pd.get_dummies(df, columns=cat_features, drop_first=True)

        # Align columns με το scaler
        expected_features = scaler.feature_names_in_
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_features]

        # Scaling
        X_scaled = scaler.transform(df)

        # Prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        label = "Dropout" if prediction == 1 else "No Dropout"
        return f"Prediction: {label} (probability of dropout: {probability:.1%})"

    except Exception as e:
        return f"Error: {str(e)}"

@tool
def retrieve_information(query: str) -> str:
    """
    Ανακτά πληροφορίες από τη βάση γνώσης για ερωτήσεις σχετικές 
    με το student dropout.
    """
    from src.rag import retrieve
    return retrieve(query)

@tool
def dataset_statistics(column_name: str) -> str:
    """
    Returns summary statistics for any column of the student dropout dataset.
    Input: column name as a string (e.g. 'GPA', 'Attendance_Rate', 'Stress_Index', 'Gender').
    For numerical columns it returns mean, median, std, min and max, overall and split by dropout status.
    For categorical columns (e.g. Gender) it returns the percentage of each category
    and the dropout rate within each category.
    Use this when the user asks about averages, distributions, percentages or class balance in the dataset.
    """
    try:
        df = pd.read_csv("data/dataset.csv")

        # Handle unknown column names
        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available columns: {', '.join(df.columns)}"

        col = df[column_name]
        dropout_mask = df["Dropout"] == 1

        # Categorical column: category percentages and dropout rate per category
        if not pd.api.types.is_numeric_dtype(col):
            shares = col.value_counts(normalize=True) * 100
            dropout_rates = df.groupby(column_name)["Dropout"].mean() * 100
            lines = [f"Distribution of '{column_name}' (n={len(df)}):"]
            for category, pct in shares.items():
                lines.append(
                    f"- {category}: {pct:.1f}% of students, dropout rate {dropout_rates[category]:.1f}%"
                )
            lines.append(f"- Overall dropout rate in dataset: {dropout_mask.mean():.1%}")
            return "\n".join(lines)

        # Numerical column: summary statistics overall and by dropout status
        return f"""Statistics for '{column_name}':
- Overall: mean={col.mean():.2f}, median={col.median():.2f}, std={col.std():.2f}, min={col.min():.2f}, max={col.max():.2f}
- Dropout students: mean={col[dropout_mask].mean():.2f}, median={col[dropout_mask].median():.2f}
- Non-dropout students: mean={col[~dropout_mask].mean():.2f}, median={col[~dropout_mask].median():.2f}
- Overall dropout rate in dataset: {dropout_mask.mean():.1%}"""

    except Exception as e:
        return f"Error: {str(e)}"