import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def kmeans_and_logistic_regression(file_path, k, fitur1_index=1, fitur2_index=0):
    file = pd.read_csv(file_path)

    for column in file.columns:
        if file[column].dtype == type(object) or len(set(file[column])) <= 10:
            le = preprocessing.LabelEncoder()
            file[column] = le.fit_transform(file[column])

    feature_columns = file.columns.tolist()

    X = file[feature_columns]

    X = X.values

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    labels= kmeans.predict(X)
    centers = kmeans.cluster_centers_

    kerekatan = silhouette_score(X, labels)

    X_fitur = X
    y_label = labels

    X_train, X_test, y_train, y_test = train_test_split(X_fitur, y_label, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    akurasi = accuracy_score(y_test, y_pred)

    plt.figure(figsize=(10, 6))
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    scatter = plt.scatter(X[:, fitur1_index], X[:, fitur2_index], c=labels, s=50, cmap='viridis', edgecolors='black', linewidth=1, alpha=0.75)
    plt.scatter(centers[:, fitur1_index], centers[:, fitur2_index], c='red', s=200, alpha=0.75, edgecolors='black')
    plt.xlabel(feature_columns[fitur1_index], fontsize=12)
    plt.ylabel(feature_columns[fitur2_index], fontsize=12)
    plt.title(f'K-Means Clustering (n_clusters={k})', fontsize=14)
    plt.colorbar(scatter)
    namefiletime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    path = 'img/kmeans'+namefiletime+'.png'
    plt.savefig('static/'+path, transparent=True)
    kerekatan = int(kerekatan*100)
    akurasi = int(akurasi*100)
    return {'akurasi':akurasi, 'kerekatan':kerekatan, 'path':path}