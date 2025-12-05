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
