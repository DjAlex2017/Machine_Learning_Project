import os
from utils.kaggle_loader import load_kaggle
from utils.mendeley_loader import load_mendeley
from utils.uci_loader import load_uci
from utils.preprocessing import scale_split
from utils.evaluation import evaluate
from utils.clustering import plot_tsne_true_labels, plot_tsne_kmeans
from utils.eda import eda_plots

from analysis.feature_importance import plot_feature_importance

from models.logistic_model import train_logistic
from models.decision_tree_model import train_decision_tree
from models.random_forest_model import train_random_forest
from models.svm_model import train_svm

def process_dataset(name, load_fn):
    print(f"\n##################################################")
    print(f"PROCESS DATASET: {name}")
    print(f"##################################################\n")

    X, y = load_fn()

    # 1. EDA (Run once per dataset)
    print(">>> Running EDA...")
    eda_plots(X.join(y), name=name, save_dir="plots")

    # 2. Clustering + t-SNE (Run once per dataset)
    print(">>> Running t-SNE...")
    plot_tsne_true_labels(X, y, title=f"{name} - t-SNE (True Labels)", 
                          save_path=f"plots/{name.replace(' ', '_')}_tsne.png")

    # 3. Preprocess + split
    X_train, X_test, y_train, y_test, pipe = scale_split(X, y)

    # Get feature names from preprocessor
    try:
        if hasattr(pipe.named_steps['prep'], 'get_feature_names_out'):
             feature_names = pipe.named_steps['prep'].get_feature_names_out()
        else:
             feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
    except:
        # Fallback if extraction fails
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    # 4. Run All Models
    models_to_run = ["tree", "forest", "svm", "logistic"]
    
    for model_type in models_to_run:
        print(f"\n--------------------------------------------------")
        print(f"Running Model: {model_type.upper()} on {name}")
        print(f"--------------------------------------------------")

        # Train
        if model_type == "forest":
            model = train_random_forest(X_train, y_train)
        elif model_type == "svm":
            model = train_svm(X_train, y_train)
        elif model_type == "logistic":
            model = train_logistic(X_train, y_train)
        else:  # default to tree
            model = train_decision_tree(X_train, y_train)

        # Evaluate
        evaluate(model, X_test, y_test)

        # Feature Importance
        # Support both tree-based importance and linear coefficients
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
            top_features = plot_feature_importance(
                model,
                feature_names,
                title=f"{name} {model_type.upper()} - Top 10 Features",
                save_path=f"plots/{name.replace(' ', '_')}_{model_type}_importance.png"
            )
            if top_features is not None:
                print("Top Features:\n", top_features)
        else:
            print(f"Skipping feature importance for {model_type} (not supported).")


def main():
    if not os.path.exists("plots"):
        os.makedirs("plots")

    process_dataset("UCI Dataset", load_uci)
    process_dataset("Kaggle Exams", load_kaggle)
    process_dataset("Mendeley Dataset", load_mendeley)

if __name__ == "__main__":
    main()
