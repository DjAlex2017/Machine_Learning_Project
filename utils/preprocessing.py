import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(path, early_warning=True):
    data = pd.read_csv(path, sep=';')
    data['pass'] = (data['G3'] >= 10).astype(int)

    if early_warning:
        data = data.drop(columns=['G1', 'G2', 'G3'])
    else:
        data = data.drop(columns=['G3'])

    X = pd.get_dummies(data.drop(columns=['pass']), drop_first=True)
    y = data['pass']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train[X.select_dtypes(include='number').columns] = scaler.fit_transform(
        X_train[X.select_dtypes(include='number').columns]
    )
    X_test[X.select_dtypes(include='number').columns] = scaler.transform(
        X_test[X.select_dtypes(include='number').columns]
    )

    return X_train, X_test, y_train, y_test

def scale_split(X, y, test_size=0.2, random_state=42):
    """
    General-purpose splitter + scaler for ANY dataset (Kaggle, Mendeley, UCI, etc.)
    """

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Scale numeric features only
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test

