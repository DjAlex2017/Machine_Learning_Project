from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
    max_depth=None,          # allow full depth
    min_samples_split=10,    # prevents overfitting
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)
    model.fit(X_train, y_train)
    return model
