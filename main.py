from utils.kaggle_loader import load_kaggle
from utils.mendeley_loader import load_mendeley
from utils.uci_loader import load_uci
from utils.preprocessing import scale_split
from utils.evaluation import evaluate
from utils.clustering import plot_tsne_kmeans
from utils.eda import eda_plots
from analysis.feature_importance import plot_feature_importance
from models.logistic_model import train_logistic
from models.decision_tree_model import train_decision_tree
from models.random_forest_model import train_random_forest

from models.gradient_boosting_model import train_gradient_boosting
from models.svm_model import train_svm
from models.kmeans_model import train_kmeans

def run_dataset(name, load_fn, model_type="decision_tree"):
    print(f"\n==============================")
    print(f"Running dataset: {name}")
    print(f"Running model: {model_type}")
    print(f"==============================\n")

    X, y = load_fn()

    # EDA
    eda_plots(X.join(y), name=name)

    # Clustering + t-SNE
    plot_tsne_kmeans(X, title=f"{name} - t-SNE + KMeans")

    # Preprocess + split
    X_train, X_test, y_train, y_test = scale_split(X, y)

    # Choose model type
    model = None
    if model_type == "random_forest":
        print("Training Random Forest...")
        model = train_random_forest(X_train, y_train)
    elif model_type == "decision_tree":
        print("Training Decision Tree...")
        model = train_decision_tree(X_train, y_train)
    elif model_type == "logistic":
        print("Training Logistic Regression...")
        model = train_logistic(X_train, y_train)
    elif model_type == "gradient_boosting":
        print("Training Gradient Boosting...")
        model = train_gradient_boosting(X_train, y_train)
    elif model_type == "svm":
        print("Training SVM...")
        model = train_svm(X_train, y_train)
    elif model_type == "kmeans":
        print("Training KMeans (Unsupervised)...")
        model = train_kmeans(X_train)
        # KMeans doesn't have predict like supervised models for evaluation in the same way
        # We can just return here or adapt evaluation
        print("KMeans training complete.")
        return

    # Evaluate
    evaluate(model, X_test, y_test)

    # Feature Importance (Tree-based models only usually)
    if model_type in ["random_forest", "decision_tree", "gradient_boosting"]:
        top_features = plot_feature_importance(
            model,
            X_train.columns,
            title=f"{name} ({model_type}) - Top 10 Important Features"
        )
        print("Top Features:\n", top_features)

def main():
    datasets = [
        ("UCI Dataset", load_uci),
        ("Kaggle Exams", load_kaggle),
        ("Mendeley Dataset", load_mendeley)
    ]

    models = [
        "logistic",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "svm",
        "kmeans"
    ]

    for data_name, load_fn in datasets:
        print(f"\n##################################################")
        print(f" PROCESSING DATASET: {data_name}")
        print(f"##################################################\n")
        
        # Run all models on this dataset
        for model in models:
            try:
                run_dataset(data_name, load_fn, model_type=model)
            except Exception as e:
                print(f"Error running {model} on {data_name}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("error.log", "w") as f:
            f.write(traceback.format_exc())
        print("An error occurred. Check error.log")
