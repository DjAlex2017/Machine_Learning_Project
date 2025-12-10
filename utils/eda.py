import seaborn as sns
import matplotlib.pyplot as plt

def eda_plots(df, name="dataset", target="pass"):
    # Histogram
    plt.figure()
    df[target].value_counts().plot(kind='bar')
    plt.title(f"Target distribution - {name}")
    plt.savefig(f"plots/{name}_target_dist.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title(f"Correlation Heatmap - {name}")
    plt.savefig(f"plots/{name}_correlation.png")
    plt.close()
