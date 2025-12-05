from utils.kaggle_loader import load_kaggle
from utils.mendeley_loader import load_mendeley
# from utils.uci_loader import load_uci
from utils.preprocessing import scale_split
from utils.evaluation import evaluate
from utils.clustering import plot_tsne_kmeans
from utils.eda import eda_plots

from analysis.feature_importance import plot_feature_importance

from models.logistic_model import train_logistic
from models.decision_tree_model import train_decision_tree
from models.random_forest_model import train_random_forest

def run_dataset(name, load_fn, use_complex=False):
    print(f"\n==============================")
    print(f"Running dataset: {name}")
    print(f"==============================\n")

    X, y = load_fn()

    # EDA
    eda_plots(X.join(y))

    # Clustering + t-SNE
    plot_tsne_kmeans(X, title=f"{name} - t-SNE + KMeans")

    # Preprocess + split
    X_train, X_test, y_train, y_test = scale_split(X, y)

    # Choose model type
    if use_complex:
        print("Using COMPLEX model: Random Forest")
        model = train_random_forest(X_train, y_train)
    else:
        print("Using SIMPLE model: Decision Tree")
        model = train_decision_tree(X_train, y_train)

    # Evaluate
    evaluate(model, X_test, y_test)

    # Feature Importance
    top_features = plot_feature_importance(
        model,
        X_train.columns,
        title=f"{name} - Top 10 Important Features"
    )
    print("Top Features:\n", top_features)

    

def main():
    #run_dataset("UCI Dataset", load_uci, use_complex=False)
    run_dataset("Kaggle Exams", load_kaggle, use_complex=False)
    run_dataset("Mendeley Dataset", load_mendeley, use_complex=True)

if __name__ == "__main__":
    main()
