import seaborn as sns
import matplotlib.pyplot as plt

def eda_plots(df, target="pass"):
    # Histogram
    df[target].value_counts().plot(kind='bar')
    plt.title("Target distribution")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
