from sklearn.cluster import KMeans

def train_kmeans(X_train):
    model = KMeans(n_clusters=3, random_state=42, n_init="auto")
    model.fit(X_train)
    return model
