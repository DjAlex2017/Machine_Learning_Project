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
