from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    cm = confusion_matrix(y_test, y_pred)
    return report, auc, cm
