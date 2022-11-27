import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('./dataset.csv')

x = df['Horário']
y = df['Idade']

data = list(zip(x,y))


sse = []
for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.savefig("Elbow method")