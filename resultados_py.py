import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simple tests
print('Hello world!')
17+25

# Print dataframe
pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

# Show plot
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()


##Apriori, sin realizar la normalizacion podemos conocer los bancos con el mayor numero de transacciones
## en nuestro data set



##Estadisticas descriptivas
ruta = 'C:/Users/Victor/Documents/18semestre/EntrevistasdeTrabajo/AplicacionENTROPIA/test_entropia/'
df = pd.read_csv(ruta+'Consumer_Complaints.csv')
##especificamos las columnas de
df.head()


df =

min(pd.to_datetime (df['Date received']))

#Pudiera ser de interes conocer como se ha comportado el servicio de los bancos
#a traves de los 5 años de registro que contamos en el sentido de si las quejas
# o comentarios de los clientes han sido resueltas

#Mas importante aun la comparacion entre un banco de interes vs el resto, para propositos
#ilustrativos nosotros graficamos los dos bancos con mayor numero de transacciones



df.dtypes
np.unique(df['Product'])


nombre_bancos, conteo_nombres = np.unique(df['Company'],return_counts=True)

orden_desc = conteo_nombres.argsort()[::-1]

nombre_bancos[orden_desc][:10]

conteo_nombres[orden_desc][:10]

np.quantile(conteo_nombres,0.95)
np.sum(conteo_nombres[orden_desc]>235.8)


[1,2,3,4][::-1]


#Pensando que la mayoria de los bancos


len(np.unique(df['Company']))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


print("rocks :", lemmatizer.lemmatize("rocks"))
