import pandas as pd

def load_uci():
    # Helper to load UCI dataset
    df = pd.read_csv("data/student-mat.csv", sep=";")

    # Logic: G3 >= 10 is a pass
    df["pass"] = (df["G3"] >= 10).astype(int)

    # Drop G1, G2, G3 to avoid data leakage (predicting grade from grade)
    df = df.drop(columns=["G1", "G2", "G3"])

    X = df.drop(columns=["pass"])
    y = df["pass"]

    # One-hot encoding for strings
    X = pd.get_dummies(X, drop_first=True)

    return X, y
