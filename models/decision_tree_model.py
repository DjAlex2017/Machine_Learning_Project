from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model
