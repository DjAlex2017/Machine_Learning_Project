import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
#  SAFE UCI LOADER (NO ENCODING, NO SCALING, NO LEAKAGE)
# ------------------------------------------------------------

def load_and_prepare_data(path, early_warning=True):
    """
    Loads the UCI dataset from CSV.
    Creates pass/fail label.
    Removes grade columns to prevent leakage.
    Returns raw X and y (NO scaling or encoding here).
    """

    data = pd.read_csv(path, sep=';')

    # Create target
    data['pass'] = (data['G3'] >= 10).astype(int)

    # Drop grade columns to avoid leakage
    if early_warning:
        data = data.drop(columns=['G1', 'G2', 'G3'])
    else:
        data = data.drop(columns=['G3'])

    X = data.drop(columns=['pass'])
    y = data['pass']

    return X, y


# ------------------------------------------------------------
#  CLEAN, LEAK-PROOF TRAIN/TEST SPLIT + PREPROCESSOR
# ------------------------------------------------------------

def scale_split(X, y, test_size=0.2, random_state=42):
    """
    General-purpose splitter + scaler + encoder.
    Handles ANY dataset safely (Kaggle, UCI, Mendeley, custom).
    """

    # Split first (NO preprocessing before split!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Identify numeric + categorical columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

    # Preprocessor: scale numeric, one-hot encode categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Build preprocessing pipeline
    pipe = Pipeline([
        ('prep', preprocessor)
    ])

    # Fit ONLY on training data
    import warnings
    X_train_processed = pipe.fit_transform(X_train)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*unknown categories.*")
        X_test_processed = pipe.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, pipe
