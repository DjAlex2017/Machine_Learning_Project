from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

    model.fit(X_train, y_train)
    return model
