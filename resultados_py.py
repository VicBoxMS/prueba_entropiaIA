#Librerias a utilizar

#Puesto que queremos un trabajo lo mas limpio posible, no imprimiremos los # WARNING:
import warnings
warnings.filterwarnings("ignore")

#Para los incisos a) b) c) d)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#incisos e)

#inciso f)

#Convertir una variable categorica en numerica
def bin_tiempo_respuesta(x):
    #Una respuesta oportuna
    if x=='Yes':
        return 1
    #Una respuesta no oportuna
    elif x=='No':
        return 0
    #A manera de validar posibles errores
    else:
        return -1

#Funcion utilizada para responder el inciso c)
def correccion_nombres_bancos(x):
    distancia_inicial = 10000
    for i in top_quantil_bancos:
        distancia = edit_distance(lemmatizer.lemmatize(x),lemmatizer.lemmatize(i))
        if distancia < distancia_inicial:
            print(distancia)
            distancia_inicial=distancia
            correccion = i
    return correccion

##Apriori, sin realizar la normalizacion podemos conocer los bancos con el mayor numero de transacciones
## en nuestro data set, bajo el argumento de que

##Estadisticas descriptivas

#Lectura del dataset
ruta = 'C:/Users/Victor/Documents/18semestre/EntrevistasdeTrabajo/AplicacionENTROPIA/test_entropia/'
df = pd.read_csv(ruta+'Consumer_Complaints.csv')
df.head()
len(df)

df['Timely response?'] = df['Timely response?'].apply(bin_tiempo_respuesta)
df['Date received'] =pd.to_datetime(df['Date received'])

#Agrupamos por mes y año
periodo_mensual = df['Date received'].dt.to_period("M")
tiempos_de_respuesta = df.groupby(periodo_mensual)['Timely response?'].agg(['sum', 'mean'])
tiempos_de_respuesta = tiempos_de_respuesta.reset_index()

x1 = tiempos_de_respuesta['Date received']
x1 = x1.dt.to_timestamp()
y1 = tiempos_de_respuesta['mean']
media_global = df['Timely response?'].mean()
media_global
plt.scatter(x1,y1)
plt.hlines(media_global,min(x1),max(x1))
plt.show()

#Los tiempos oportunos de solución eran ligeramente
#mejor en el año 2013,2014 dicho de otra forma,
#Tambien podemos comparar lo anterior, entre las
#dos instituciones financieras con el mayor numero
#de transacciones

#Otra estadistica relevante que podemos concluir del
#grafico anterior es el hecho de que en los ultimos años
#menos del 5% de los clientes piensa que su respuesta
#no fue atendida en un tiempo oportuno.

"""
#Convertimos los objetos a un formato de fechas
#y posteriormente pasamos al formato dd/mm/yyyy
#df['Date received'] = pd.to_datetime(df['Date received']).dt.strftime('%d/%m/%Y')
"""

##Comparación entre la compañia con el mayor numero de transacciones
##y la comparacion con una compañia que ocupa el puesto numero 20
##en el numero de transacciones

df_america = df[df['Company']=='Bank of America']
df_america = df_america.reset_index()

df_nationstar = df[df['Company']=='Nationstar Mortgage']
df_nationstar = df_nationstar.reset_index()

#Agrupamos por mes y año para el banco de america
periodo_mensual = df_america['Date received'].dt.to_period("M")
tiempos_de_respuesta = df_america.groupby(periodo_mensual)['Timely response?'].agg(['sum', 'mean'])
tiempos_de_respuesta = tiempos_de_respuesta.reset_index()
x1 = tiempos_de_respuesta['Date received']
x1 = x1.dt.to_timestamp()
y1 = tiempos_de_respuesta['mean']
media1 = df_america['Timely response?'].mean()

#Agrupamos por mes y año para Nationstar
periodo_mensual = df_nationstar['Date received'].dt.to_period("M")
tiempos_de_respuesta = df_nationstar.groupby(periodo_mensual)['Timely response?'].agg(['sum', 'mean'])
tiempos_de_respuesta = tiempos_de_respuesta.reset_index()
x2 = tiempos_de_respuesta['Date received']
x2 = x2.dt.to_timestamp()
y2 = tiempos_de_respuesta['mean']
media2 = df_nationstar['Timely response?'].mean()

###Mostraremos la puntuación promedio por año comparar
###para cada banco en este sentido
plt.scatter(x1,y1)
plt.scatter(x2,y2,color='orange')
plt.hlines(media1,min(x1),max(x1))
plt.hlines(media2,min(x1),max(x1),color='orange')
plt.show()

#Agrupar por año y por compañida


##
nombre_bancos, conteo_nombres = np.unique(df['Company'],return_counts=True)
orden_desc = conteo_nombres.argsort()[::-1]
top10_transacciones_bancos = nombre_bancos[orden_desc][:20]


#bancos diferentes identificados
len(nombre_bancos) #3933 bancos diferentes

np.sum(conteo_nombres[orden_desc][::-1]<50)
quantil_interes = np.quantile(conteo_nombres,0.975)
np.sum(conteo_nombres[orden_desc]>quantil_interes)

#
titles = top10_transacciones_bancos
df_top10 = df.query("Company in @titles")
resumen = df_top10.groupby('Company')['Timely response?'].agg(['count', 'mean'])
resumen.sort_values(by='count',ascending=False)
resumen.corr()

#Si consideramos los 25 bancos con el mayor numero de transacciones
#podemos notar que a menor de numero de transacciones los bancos pueden
#resolver en tiempo las inquietudes reportadas, lo que se traduce en una taza
#de respuesta oportuna mas alta.





#Pudiera ser de interes conocer como se ha comportado el servicio de los bancos
#a traves de los 5 años de registro que contamos en el sentido de si las quejas
# o comentarios de los clientes han sido resueltas

#Mas importante aun la comparacion entre un banco de interes vs el resto, para propositos
#ilustrativos nosotros graficamos los dos bancos con mayor numero de transacciones


#utilizamos query para filtrar solo las compañias de interes
#las compañias de interes son aquellas que tiene un mayor numero de transacciones

#Pensando que la mayoria de los bancos
len(np.unique(df['Company']))

import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet
print("rocks :", lemmatizer.lemmatize("rocks"))

#Tomado de https://www.nltk.org/_modules/nltk/metrics/distance.html

from nltk.metrics.distance import edit_distance

#Realizamos el conteo de instituciones/bancos
nombre_bancos, conteo_nombres = np.unique(df['Company'],return_counts=True)
len(nombre_bancos) #3933 bancos diferentes

#Por el numero de transacciones, realizamos un corte en el quantil 0.975
#lo que es igual a un valor approximado de 630
quantil_interes = np.quantile(conteo_nombres,0.975)

#Asumiremos que si una institucion tiene mas de 630 registros, el nombre está
#escrito de manera correcta, por lo que aquellas instituciones que tienen

#Ordenamos por numero de registros
orden_desc = conteo_nombres.argsort()[::-1]
num_bancos_quantil = int(len(nombre_bancos)*0.025)
top_quantil_bancos = nombre_bancos[orden_desc][:num_bancos_quantil]
top_quantil_bancos_lematizados = [lemmatizer.lemmatize(j) for j in top_quantil_bancos]
#Creamos una funcion para corregir el nombre de los bancos a partir del quantil
#0.025 que hemos corregido


#n_bancos almacena el numero de bancos que estan en el quantil
n_bancos = len(top_quantil_bancos)
#conteo almacena el numero de transacciones al considerar n_bancos, tenemos 561471 transacciones
conteo = sum(conteo_nombres[orden_desc][:n_bancos])
conteo

df['Company'] = df['Company'].apply(correccion_nombres_bancos)

nombre_bancos, conteo_nombres = np.unique(df['Company'],return_counts=True)




#1+1

#Mapa de donde provienen las quejas


#¿Variable para predecir el tipo de Producto?

#'Submitted via' ['Phone', 'Web', 'Fax', 'Referral', 'Postal mail', 'Email']
np.unique(df['Submitted via'],return_counts=True)

#¿Como ha evolucianado el medio de comunicación entre el usuario y el banco?

df_cruza = pd.crosstab(df['Date received'],df['Submitted via']).reset_index()
periodo_anual = df_cruza['Date received'].dt.to_period("Y")
#df_cruza.groupby(periodo_anual).sum().sum()
df_cruza.groupby(periodo_anual).sum().apply(lambda r : r/r.sum(),axis=1)
#Independientemente del año que se quiera considerar,
#vemos que la mayoria de las situaciones reportadas
#ocurren por medio de la web. Podemos ver que en los ultimos años registrados
#la comunicación via internet ha tomado mas terreno y la que ha perdido mas impulso
# es la etiquetada como 'Referral'. El e-mail por su parte es quien tiene la menor
#participación para el año 2016 con un porcentaje de $0.000048*100=0.0048/%$.


#Tiempo de respuesta ¿?
plt.hist(df['Timely response?'])

#Nos damos cuenta que la mayoria de las quejas se resuelven en un día, sin embargo
# algunas quejas parecen resolverse inmediatamente, en ese sentido.
#¿Cuales son los temas que se resuelven mas rapido?

df_temas_x_dias = pd.crosstab(index=df['Product'], columns=df['Timely response?'])
#,0:'0 dias',
df_temas_x_dias=df_temas_x_dias.apply(lambda r: r/r.sum(),axis=1)#.reset_index()
df_temas_x_dias = df_temas_x_dias.rename(columns={0:'0 dias',1:'1 dia'})
df_temas_x_dias.sort_values(by='0 dias',ascending=False)

#Podemos ver que una proporcion de tramites más alta que puede ser
#resuelta en un día para los tramites Moneda virtual ('Virtual Currency') y
#Préstamo a corto plazo'('payday loan').









#¿De que tema hablan las quejas?
productos , conteo_productos = np.unique(df['Product'],return_counts=True)
orden_desc = conteo_productos.argsort()[::-1]
orden_desc
df_conteo_productos = pd.DataFrame([productos[orden_desc],conteo_productos[orden_desc]]).T
df_conteo_productos.rename(columns={0:'productos',1:'conteo_productos'})

"""
Son quejas realizadas por clientes hacia productos y servicios financieros ,
algunos de los principales temas/productos son:
-Hipotecas
-Cobro de deudas
-Informes de creditos
-Tarjetas de creditos
-Cuentas bancarias o servicios
"""

#Bolsa de palabras, nube de palabras

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#stop_words contiene todas las palabras secundarias que no son de nuestro quantil_interes
#al momento de generar una bolsa de palabras/nube de palabras.

stop_words = stopwords.words('english')
stop_words.append('XXXX')
stop_words.append('xxxx')
nonan = ~df['Consumer complaint narrative'].isna()
comentarios_clientes = df['Consumer complaint narrative'][nonan]
comentarios_clientes_cortados = [i[:500] for i in comentarios_clientes]

comentarios_unidos = comentarios_clientes_cortados[0]+comentarios_clientes_cortados[1]
for i in range(2,100):
  comentarios_unidos+=comentarios_clientes_cortados[i]

wordcloud = WordCloud().generate(comentarios_unidos)
# imagen generada a partir de toda la informacion
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#Podemos ver que en conjunto los textos hablas sobre tarjetas, creditos, tasas de
#amortizacion, ademas de que por cuestiones de confidencialidad hay mucha
#informacion anonimizada.

#Otra perspectiva que pudieramos tener de los datos es obtener una bolsa de palabras
#por cada tema.





#Desarrolla un modelo de machine learning que prediga el tipo de producto con base en el
#texto. Puedes usar la técnica que desees, pero debes explicar qué métricas de evaluación usas
#para determinar tu elección del model

#Recordamos de la pregunta d) que tenemos 12 temas y estamos frente a un conjunto de
#datos no equilibrado, por lo que la metrica 'accuracy' no es una opción.
#Por el contrario optaremos por 'f1-score' y puesto que estamos ante un escenario
#de clasificación multiclase, especificamente 'f1-score-macro'.



from sklearn.model_selection import train_test_split

#select_dtypes(['object'])

for col in df.columns:
    print(col,':',df[col].nunique(),':',df[col].isna().sum())



df_preprocesado = df
y = df_preprocesado['Product']

#Para esta parte, decidimos utilizar la columna Codigo Postal ('Zip Code') que corresponde al
#codigo postal donde se realiza el reporte, la información puede ser utilizada
#para sustituir o complementar la variable Estado ('State') ya que al clusterizar
#podemos encontrar similitudes entre algunos estados o estudiar mejor la frontera
#que existe entre ellos.

#funcion que sustituye los digitos no reportados en el zip como 'X' por el valor 0
def sustituir_x_zip(x):
    parcial=''
    for i in range(len(str(x))):
        try:
            int(x[i])
            parcial+=x[i]
        except:
            parcial+='0'
    return int(parcial)




df_tsne = df[['State','ZIP code']][~df['State'].isna()]
df_tsne = df_tsne.reset_index()
df_tsne['ZIP code'] = df_tsne['ZIP code'].apply(sustituir_x_zip)
estados = df_tsne['State']
X = np.array(df_tsne['ZIP code'])
X = X.reshape(-1,1)
X = X[:10000]
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
estados = estados[:10000]

len(np.unique(estados))

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing  import LabelEncoder
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd


n_a_graficar = 5000
etiquetas_estados = LabelEncoder().fit_transform(df_tsne['State'])
X_embedded = TSNE(n_components=2,init='random').fit_transform(X)
plt.scatter(X_embedded[:n_a_graficar,0],X_embedded[:n_a_graficar,1])
plt.show()

plt.scatter(X_embedded[:n_a_graficar,0],X_embedded[:n_a_graficar,1],c=etiquetas_estados[:n_a_graficar])
plt.show()

##A partir de la tecnica t-sne podemos ver una cierta agrupación de los codigos postales,
##que de hecho coincide de manera parcial con los estados, podemos ver segmentos de
#puntos del mismo color, lo que hace referencia a que pertenecen a un unico estado.

#Ahora bien, sobre dicha proyeccion realizada, lo ideal sería definir un numero
#de clusters y estudiar la posibilidad de utilizar los clusters generados como
#una variable predictora


from sklearn.cluster import DBSCAN
db = DBSCAN(eps=2, min_samples=20).fit(X_embedded)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

from sklearn import metrics
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

plt.scatter(X_embedded[:,0],X_embedded[:,1],c=labels)


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    class_member_mask = labels == k
    xy = X_embedded[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X_embedded[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()

#Esta es la agrupacion que sugiere el algoritmo dbscan



import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


# Create a graph capturing local connectivity. Larger number of neighbors
# will give more homogeneous clusters to the cost of computation
# time. A very large number of neighbors gives more evenly distributed
# cluster sizes, but may not impose the local manifold structure of
# the data

#Para no definir Apriori un numero de clusters, podriamos ocupar
#un algoritmo DBSCAN, pero es comun que los clusters que se formen de este algoritmo
#sean circulares, almenos para un espacio bidimensional, los resultados de este
#algoritmo se omiten en el presente reporte

#Por lo que podemos ver de la tecnica t-sne, estariamos pensando en que existe una
# conexion entre elementos del cluster, por lo que haremos un grafo que nos ayude
# a entender las conexión y aplicaremos un cluster aglomerativo a t-sne considerando
#las distancias de vecinos mas cercanos consideraremos conexiones completas y simples
#Propondremos


knn_graph = kneighbors_graph(X_embedded, 30, include_self=False)
model = AgglomerativeClustering(
    linkage=linkage, connectivity=connectivity, n_clusters=n_clusters
)
t0 = time.time()
model.fit(X_embedded)
elapsed_time = time.time() - t0
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=model.labels_, cmap=plt.cm.nipy_spectral)
plt.title(
    "linkage=%s\n(time %.2fs)" % (linkage, elapsed_time),
    fontdict=dict(verticalalignment="top"),
)
plt.axis("equal")
plt.axis("off")



print("Adjusted Rand Index: %0.3f" % ari)
print("Adjusted Mutual Information: %0.3f"% ami)




knn_graph = kneighbors_graph(X_embedded, 30, include_self=False)
for connectivity in (None,knn_graph):
    for n_clusters in (30, 50):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(("complete", "single")):
            plt.subplot(1, 4, index + 1)
            model = AgglomerativeClustering(
                linkage=linkage, connectivity=connectivity, n_clusters=n_clusters
            )
            t0 = time.time()
            model.fit(X_embedded)
            elapsed_time = time.time() - t0
            ari = metrics.adjusted_rand_score(etiquetas_estados[:len(model.labels_)], model.labels_)
            ami = metrics.adjusted_mutual_info_score(etiquetas_estados[:len(model.labels_)], model.labels_)
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=model.labels_, cmap=plt.cm.nipy_spectral)
            plt.title(
                "linkage=%s\n ARI : %.2fs ; AMI: %.2fs" % (linkage,ari,ami),
                fontdict=dict(verticalalignment="top"),
            )
            plt.axis("equal")
            plt.axis("off")
            plt.subplots_adjust(bottom=0, top=0.83, wspace=0, left=0, right=1)
            plt.suptitle(
                "n_cluster=%i, connectivity=%r"
                % (n_clusters, connectivity is not None),
                size=14,
            )
plt.show()


#De acuerdo a las metricas ari y ami, y bajo el supuesto de que atraves del codigo
#postal, podemos encontrar clusters que represen a los estados, los mejores modelos
#de clusterización parecen ser los que consideran un enlace simple ('single')
#independientemente de si considere o no la conectividad (connectivity)





#Por cuestiones de tiempo evitaremos modelos como bag-of-words y tfdi.
#y apostaremos directamente por modelos de transformers, utilizaremos un
#modelo previamente entrenado para crear los embeddings y tambien utilizaremos
#ajuste fino.
#Por las capacidades de mi computadora, esta ultima parte, la haremos directamente
#en google collab.

##Utilizando el modelo distilbert-base-uncased
## para realizar ingenieria de caracteristicas (extraccion de caracteristicas)
## y posteriormente ingresar a 2 modelos de machine learning regresión logistica
# y bosques aleatorios, ambos elegidos por la rapidez con la que se puede entrenar
#un modelo



##



##Evaluacion del modelo
