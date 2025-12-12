from sklearn.linear_model import LogisticRegression

def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=5000, class_weight='balanced', C=1.0, solver='liblinear')

    model.fit(X_train, y_train)
    return model
