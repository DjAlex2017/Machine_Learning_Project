import pandas as pd

def load_kaggle():
    df = pd.read_csv("data/StudentsPerformance.csv")

    # Convert target into pass/fail like UCI
    df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    df["pass"] = (df["average_score"] >= 70).astype(int)

    df = df.drop(columns=["average_score"])

    X = df.drop(columns=["pass"])
    y = df["pass"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y