from sklearn.svm import SVC

def train_svm(X_train, y_train):
    """
    Trains a Support Vector Machine (SVM) model with tuned regularization.
    Works better for your noisy and imbalanced datasets.
    """
    model = SVC(
        kernel='rbf',
        C=2,                # slightly stronger regularization than default
        gamma='scale',      # auto-adjusts to dataset size
        probability=True,
        class_weight='balanced',   # FIXES imbalance issues (VERY IMPORTANT)
        random_state=42
    )

    model.fit(X_train, y_train)
    return model
