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


jovemTotalAcidentes = 0
jovemAdultoTotalAcidentes = 0
adultoTotalAcidentes = 0
idosoTotalAcidentes = 0

jovemTotalVelocidade = 0
jovemAdultoTotalVelocidade = 0
adultoTotalVelocidade = 0
idosoTotalVelocidade = 0

for index, row in df.iterrows():
    if(row['Idade'] > 0 and row['Idade'] <= 18):
        jovemTotalAcidentes = jovemTotalAcidentes + 1
        jovemTotalVelocidade = jovemTotalVelocidade + row['VelocidadeMaxima']
    
    elif(row['Idade'] > 18 and row['Idade'] <= 24):
        jovemAdultoTotalAcidentes = jovemAdultoTotalAcidentes + 1
        jovemAdultoTotalVelocidade = jovemAdultoTotalVelocidade + row['VelocidadeMaxima']
    
    elif(row['Idade'] > 25 and row['Idade'] <= 59):
        adultoTotalAcidentes = adultoTotalAcidentes + 1
        adultoTotalVelocidade = adultoTotalVelocidade + row['VelocidadeMaxima']

    else:
        idosoTotalAcidentes = idosoTotalAcidentes + 1
        idosoTotalVelocidade = idosoTotalVelocidade + row['VelocidadeMaxima']

novoDf = {'jovem': [jovemTotalAcidentes, (jovemTotalVelocidade / jovemTotalAcidentes)], 'jovemAdulto': [jovemAdultoTotalAcidentes, (jovemAdultoTotalVelocidade / jovemAdultoTotalAcidentes)], 'adulto': [adultoTotalAcidentes, (adultoTotalVelocidade / adultoTotalAcidentes)], 'idoso': [idosoTotalAcidentes, (idosoTotalVelocidade / idosoTotalAcidentes)]}

df = pd.DataFrame(novoDf).T.rename(columns={0: 'Total Acidentes', 1: 'Média Velocidade Máxima'})

print(df)

x = (np.array(df['Total Acidentes']))
y = (np.array(df['Média Velocidade Máxima']))

pals = ['Jovem', 'Jovem adulto', 'Adulto']#, 'Idoso']

fig, ax = plt.subplots()

ax.scatter(x, y)
ax.xlabel("Total Acidentes")
ax.ylabel("Média Velocidade Máxima")
ax.annotate("Jovem", (x[0], y[0]), xytext=(x[0]+0.05, y[0]+0.3)
'''ax.annotate("Jovem adulto", (x[1], y[1]), xytext=(x[1]+0.05, y[1]+0.3)
ax.annotate("Adulto", (x[2], y[2]), xytext=(x[2]+0.05, y[2]+0.3)
ax.annotate("Idoso", (x[3], y[3]), xytext=(x[3]+0.05, y[3]+0.3)'''
plt.savefig('Gráfico Dispersão Original')
plt.close()

