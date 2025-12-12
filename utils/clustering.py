import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def plot_tsne_kmeans(X, title="t-SNE + KMeans", n_clusters=3):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)

    X_embedded = tsne.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_embedded)

    plt.figure(figsize=(8,6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar()
    plt.show()

def plot_tsne_true_labels(X, y, title="t-SNE - True Labels", save_path=None):
    # Calculate t-SNE embedding
    # Ensure numeric input by one-hot encoding if necessary
    import pandas as pd
    X_numeric = pd.get_dummies(X, drop_first=True)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_numeric)

    plt.figure(figsize=(8, 6))

    # Define colors: Pass (1) -> Blue, Fail (0) -> Red
    # Create mask for plotting separately to handle legend easily
    pass_mask = (y == 1)
    fail_mask = (y == 0)

    plt.scatter(X_embedded[pass_mask, 0], X_embedded[pass_mask, 1], 
                c='blue', label='Pass', alpha=0.6)
    plt.scatter(X_embedded[fail_mask, 0], X_embedded[fail_mask, 1], 
                c='red', label='Fail', alpha=0.6)

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
