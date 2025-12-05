import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names, title):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index)
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    return top