from utils.preprocessing import load_and_prepare_data
from utils.evaluation import evaluate
from models.logistic_model import train_logistic
from models.decision_tree_model import train_decision_tree
from models.random_forest_model import train_random_forest
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data('data/student-mat.csv', early_warning=True)

    models = {
        'Logistic Regression': train_logistic(X_train, y_train),
        'Decision Tree': train_decision_tree(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
    }

    for name, model in models.items():
        print(f"\n* {name}")
        report, auc, cm = evaluate(model, X_test, y_test)
        print(report)
        print(f"AUC: {auc:.3f}")

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.show()

if __name__ == "__main__":
    main()
