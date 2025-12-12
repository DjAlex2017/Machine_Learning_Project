import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

def plot_feature_importance(model, feature_names, title, save_path=None):
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        # For linear models, coef_ is (n_classes, n_features) or (1, n_features)
        # We take absolute value to show magnitude of importance
        importances = pd.Series(np.abs(model.coef_[0]), index=feature_names)
    else:
        print(f"Model {type(model).__name__} does not expose feature importances or coefficients.")
        return None

    top = importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index)
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return top