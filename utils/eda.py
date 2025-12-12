import os
import seaborn as sns
import matplotlib.pyplot as plt

def eda_plots(df, name="dataset", save_dir="plots", target="pass"):
    """
    Generates:
    - Target distribution bar plot
    - Correlation heatmap (numeric features only)

    Works safely with raw DataFrames and avoids incorrect heatmaps 
    on encoded or scaled data.
    """

    # Ensure the plots directory exists
    os.makedirs("plots", exist_ok=True)

    # -----------------------------------
    # Target Distribution
    # -----------------------------------
    plt.figure(figsize=(6, 4))
    df[target].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
    plt.title(f"Target distribution - {name}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plots/{name}_target_dist.png")
    plt.close()

    # -----------------------------------
    # Correlation Heatmap (numeric only)
    # -----------------------------------
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.shape[1] > 1:  # Must have at least 2 numeric columns
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
        plt.title(f"Correlation Heatmap - {name}")
        plt.tight_layout()
        plt.savefig(f"plots/{name}_correlation.png")
        plt.close()
    else:
        print(f"[INFO] Skipped correlation heatmap for '{name}': not enough numeric columns.")
