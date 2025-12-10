from sklearn.svm import SVC

def train_svm(X_train, y_train):
    model = SVC(probability=True, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model
