import pandas as pd

def load_kaggle(path="data/StudentsPerformance.csv"):
    """
    Loads the Kaggle StudentsPerformance dataset.
    Creates a pass/fail target based on average exam score.
    Removes the raw score columns to prevent target leakage.
    
    NO one-hot encoding and NO scaling are done here.
    The preprocessing pipeline will handle that safely.
    """

    df = pd.read_csv(path)

    # Create target from exam average
    df["average_score"] = (
        df[["math score", "reading score", "writing score"]].mean(axis=1)
    )
    df["pass"] = (df["average_score"] >= 70).astype(int)

    # REMOVE columns that contain information used to compute "pass"
    # This is required to prevent leakage.
    df = df.drop(columns=[
        "average_score",
        "math score",
        "reading score",
        "writing score"
    ])

    # Separate raw features from label
    X = df.drop(columns=["pass"])
    y = df["pass"]

    # IMPORTANT:
    # Do NOT encode here.
    # Do NOT scale here.
    # The preprocessing pipeline (scale_split / ColumnTransformer)
    # will handle encoding and scaling correctly.

    return X, y
