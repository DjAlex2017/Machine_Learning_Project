import pandas as pd

def load_mendeley():
    # Load dataset
    df = pd.read_csv("data/ResearchInformation3.csv")

    # Drop missing values
    df = df.dropna()

    # Create pass/fail using GPA threshold of 3.0
    df["pass"] = (df["Overall"] >= 3.0).astype(int)

    # Drop the original target column to avoid leakage
    X = df.drop(columns=["Overall", "pass"])
    y = df["pass"]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    return X, y



