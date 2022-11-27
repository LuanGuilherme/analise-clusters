import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.figure_factory as ff

df = pd.read_csv('./dataset.csv')

masculinojovemTotalAcidentes = 0
masculinojovemAdultoTotalAcidentes = 0
masculinoadultoTotalAcidentes = 0
masculinoidosoTotalAcidentes = 0

femininojovemTotalAcidentes = 0
femininojovemAdultoTotalAcidentes = 0
femininoadultoTotalAcidentes = 0
femininoidosoTotalAcidentes = 0

masculinojovemTotalVelocidade = 0
masculinojovemAdultoTotalVelocidade = 0
masculinoadultoTotalVelocidade = 0
masculinoidosoTotalVelocidade = 0

femininojovemTotalVelocidade = 0
femininojovemAdultoTotalVelocidade = 0
femininoadultoTotalVelocidade = 0
femininoidosoTotalVelocidade = 0

for index, row in df.iterrows():
    if(row['Idade'] > 0 and row['Idade'] <= 18):
        if(row['Gênero'] == 'Male'):
            masculinojovemTotalAcidentes = masculinojovemTotalAcidentes + 1
            masculinojovemTotalVelocidade = masculinojovemTotalVelocidade + row['VelocidadeMaxima']
        else:
            femininojovemTotalAcidentes = femininojovemTotalAcidentes + 1
            femininojovemTotalVelocidade = femininojovemTotalVelocidade + row['VelocidadeMaxima']

    elif(row['Idade'] > 18 and row['Idade'] <= 24):
        if(row['Gênero'] == 'Male'):
            masculinojovemAdultoTotalAcidentes = masculinojovemAdultoTotalAcidentes + 1
            masculinojovemAdultoTotalVelocidade = masculinojovemAdultoTotalVelocidade + row['VelocidadeMaxima']
        else:
            femininojovemAdultoTotalAcidentes = femininojovemAdultoTotalAcidentes + 1
            femininojovemAdultoTotalVelocidade = femininojovemAdultoTotalVelocidade + row['VelocidadeMaxima']
    
    elif(row['Idade'] > 25 and row['Idade'] <= 59):
        if(row['Gênero'] == 'Male'):
            masculinoadultoTotalAcidentes = masculinoadultoTotalAcidentes + 1
            masculinoadultoTotalVelocidade = masculinoadultoTotalVelocidade + row['VelocidadeMaxima']
        else: 
            femininoadultoTotalAcidentes = femininoadultoTotalAcidentes + 1
            femininoadultoTotalVelocidade = femininoadultoTotalVelocidade + row['VelocidadeMaxima']

    else:
        if(row['Gênero'] == 'Male'):
            masculinoidosoTotalAcidentes = masculinoidosoTotalAcidentes + 1
            masculinoidosoTotalVelocidade = masculinoidosoTotalVelocidade + row['VelocidadeMaxima']
        else:
            femininoidosoTotalAcidentes = femininoidosoTotalAcidentes + 1
            femininoidosoTotalVelocidade = femininoidosoTotalVelocidade + row['VelocidadeMaxima']

novoDf = {'jovem masculino': [masculinojovemTotalAcidentes, (masculinojovemTotalVelocidade / masculinojovemTotalAcidentes)], 'jovem feminina': [femininojovemTotalAcidentes, (femininojovemTotalVelocidade / femininojovemTotalAcidentes)], 'jovem adulto masculino': [masculinojovemAdultoTotalAcidentes, (masculinojovemAdultoTotalVelocidade / masculinojovemAdultoTotalAcidentes)], 'jovem adulta feminina': [femininojovemAdultoTotalAcidentes, (femininojovemAdultoTotalVelocidade / femininojovemAdultoTotalAcidentes)], 'adulto masculino': [masculinoadultoTotalAcidentes, (masculinoadultoTotalVelocidade / masculinoadultoTotalAcidentes)], 'adulta feminina': [femininoadultoTotalAcidentes, (femininoadultoTotalVelocidade / femininoadultoTotalAcidentes)], 'idoso masculino': [masculinoidosoTotalAcidentes, (masculinoidosoTotalVelocidade / masculinoidosoTotalAcidentes)], 'idosa feminina': [femininoidosoTotalAcidentes, (femininoidosoTotalVelocidade / femininoidosoTotalAcidentes)] }

df = pd.DataFrame(novoDf).T.rename(columns={0: 'Total Acidentes', 1: 'Média Velocidade Máxima'})

print(df)

xOriginal = (np.array(df['Total Acidentes']))
yOriginal = (np.array(df['Média Velocidade Máxima']))

plt.scatter(xOriginal, yOriginal, s=200)
plt.xlabel("Total Acidentes")
plt.ylabel("Média Velocidade Máxima")
plt.savefig('Gráfico Dispersão Original')
plt.close()

x = stats.zscore(np.array(df['Total Acidentes']))
y = stats.zscore(np.array(df['Média Velocidade Máxima']))

print(x)
print(y)

plt.scatter(x, y, s=200)
plt.xlabel("Z-Score Total Acidentes")
plt.ylabel("Z-Score Média Velocidade Máxima")
plt.savefig('Z-Score Gráfico Dispersão')
plt.close()

print(len(df))

data = list(zip(x,y))

numClusters = len(df)

sse = []
for k in range(1, numClusters + 1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

plt.plot(range(1, numClusters + 1), sse)
plt.xticks(range(1, numClusters + 1))
plt.xlabel("Número de Clusters")
plt.ylabel("Dispersão")
plt.savefig("Curva de Elbow")
plt.close()

while numClusters > 0:
    kmeans = KMeans(n_clusters=numClusters)
    kmeans.fit(data)
    plt.scatter(x, y, c=kmeans.labels_, s=200)
    plt.savefig("Clusters - " + str(numClusters))
    plt.close()

    numClusters = numClusters - 1

fig = ff.create_dendrogram(df, orientation='left', labels=['jovem masculino', 'jovem feminina', 'jovem adulto masculino', 'jovem adulta feminina', 'adulto masculino', 'adulta feminina', 'idoso masculino', 'idosa feminina'])
fig.update_layout(width=800, height=800)
fig.write_image("Dendograma.png")
