import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime

def k_median_clustering(file_path, k, fitur1_index=0, fitur2_index=1):
    file = pd.read_csv(file_path)
    for column in file.columns:
        if file[column].dtype == type(object) or len(set(file[column])) <= 10:
            le = preprocessing.LabelEncoder()
            file[column] = le.fit_transform(file[column])
    feature_columns = file.columns.tolist()

    X = file[feature_columns]
    X = X.values

    np.random.seed(42)
    initial_indices = np.random.choice(range(len(X)), size=k, replace=False)
    centroids = X[initial_indices]

    for _ in range(100):
        distances = cdist(X, centroids, metric='cityblock')
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([np.median(X[labels == i], axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    kerekatan = silhouette_score(X, labels)

    X_fitur = X
    y_label = labels

    X_train, X_test, y_train, y_test = train_test_split(X_fitur, y_label, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    akurasi = accuracy_score(y_test, y_pred)

    
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    scatter = plt.scatter(X[:, fitur1_index], X[:, fitur2_index], c=labels, s=50, cmap='viridis', edgecolors='black', linewidth=1, alpha=0.75)
    plt.scatter(centroids[:, fitur1_index], centroids[:, fitur2_index], c='red', s=200, alpha=0.75, edgecolors='black')
    plt.xlabel(feature_columns[fitur1_index], fontsize=12)
    plt.ylabel(feature_columns[fitur2_index], fontsize=12)
    plt.title(f'K-Median Clustering (Nilai K={k})', fontsize=14)
    plt.colorbar(scatter)
    akurasi = int(akurasi*100)
    kerekatan = int(kerekatan*100)
    namefiletime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    path = 'img/kmedian'+namefiletime+'.png'
    plt.savefig('static/'+path, transparent=True)
    return {'akurasi':akurasi, 'kerekatan':kerekatan, 'path':path}


