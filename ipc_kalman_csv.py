# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 09:03:38 2018

@author: ivand
"""

# Aplicación de KF para el IPC
# Utilizando datos de Yahoo Finance
# desde enero 2017 a Diciembre 2017

##############################################################################
### Librerías necesarias para este script ###
##############################################################################
    
from pandas_datareader import data as dr # Lector de datos
import pandas as pd                      # Leer archivo CSV
import datetime                          # Manejo de fechas 
import matplotlib.pyplot as plt          # Graficador
import numpy as np                       # Análisis númerico


# A partir del 2017 Yahoo cambió su API haciendo más complicado 
# adquirir datos por medio de librerías como Pandas
# Una solución temporal es la librería fix_yahoo_finance o también
# se pueden utilizar los datos históricos bajados de yahoo en formato .csv


##############################################################################
###                   Importar datos usando Pandas                         ###
##############################################################################

# Se utiliza la herramienta read_csv de Pandas para leer el archivo de valores
# separados por comas.

ipc = pd.read_csv('ipc5y.csv', parse_dates=['Date'])

# Este archivo contiene 7 columnas que constan de:
# índice
# fecha (Date)
# valor de apertura (Open)
# valor más alto  (High)
# valor más bajo (Low)
# cierre ajustado (Adj Close)
# Volumen (Volume)

# Se checa que el tipo de variable de las columnas sea el requerido
ipc.dtypes

# Se checan los primeros 5 valores de ipc
ipc.head()

# Se reacomodan los valores por fecha
ipc = ipc.sort_values(by='Date')

# Se especifica la fecha como el índice de ipc
ipc.set_index('Date',inplace=True)

# Se gráfica el precio de cierre para comprobar que esta todo en orden
ipc['Close'].plot(figsize=(16,12))


##############################################################################
### Analisis preliminar de los datos y seleccion de series ###
##############################################################################

# Al hacer uso de Pandas y requerir datos de Yahoo Finance
# o cualquier otro servicio, se crea una estructura llamada
# Data Frame, la cual esta disenada para facilitar la
# manipulacion de los datos.

# Se puede extraer del Data Frame (df) los datos necesarios en forma
# de vectores, lo cual no es recomendable para ciertas cosas por
# ser mas tardado y crear mas lineas.

# Grafica IPC Precios de cierre ajustados
# La funcion iloc funciona para "partir" el data frame y los vectores
# en este caso nos interesa la columa 4 de Adjusted CLose
# y los renglones del 1 al 2700 (todos).

#plt.plot(ipc.iloc[1:1254, 4], label='Adj Close') 
plt.plot(ipc.iloc[1:500, 4], label='Adj Close') 

# Diferencia logaritmica entre dato t y t-1
# Para poder crear una tabla de rendimientos se puede utilizar
# la funcion apply en donde se hace uso de funciones lambda
# que ayudan a hacer de manera recursiva la instrucciones que
# se especifican, en este caso la diferencia logaritmica de los datos.
dipc = ipc.apply(lambda x: np.log(x) - np.log(x.shift(1))) 

# Revisar datos en diferencia
# La funcion head nos muestra los primeros datos del vector
dipc.head()

# Grafico de diferencias logaritmicas del IPC
plt.plot(dipc)

# Crear vector de precios de cierre ajustados
# ipc_adj = ipc.iloc[1:len(ipc),4]
# dipc_adj = dipc.iloc[1:len(dipc),4]

ipc_adj = ipc.iloc[1:500,4]
dipc_adj = dipc.iloc[1:500,4]

# Grafico de ipc_adj
plt.plot(ipc_adj, label = 'IPC Cierre Ajustado')
plt.plot(dipc_adj, label = 'IPC Cierre Ajustado Dif Logaritmicas')

# Gráfica de la variable dipc (Diferencias en los precios)
dipc.plot()
# Gráfica de la variable dipc Cierre Ajustado Diferencias en los precios
dipc_adj.plot()

# A diferencia del tratamiento de datos como vectores, se puede
# aprovechar el df, de la siguiente manera

# se crea una función de diferencias logaritmicas
ipc['Return'] = np.log(ipc['Close']/ipc['Close'].shift(1))

# se hace una grafica del precio y las diferencias (rendimientos)
ipc[['Close', 'Return']].plot(subplots=True, style='b', figsize=(8, 5))

# Para motivos de este análisis, se hace una muestra de los datos
# y se conservan los primeros 500 para la columna Return

m_ipc = ipc.iloc[1:500,6]

##############################################################################
### Estimacion de un modelo AR(a,r) ###
##############################################################################

# Para poder crear un modelo AR(p,q) es necesario conocer la naturaleza
# de la serie con la que estamos tratando. En especifico ver las relaciones
# entre los datos o su correlacion.

# Una manera visual de hacer esto es con graficas

from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot

# Si se hace una imágen de la correlación entre los valores, queda de
# la siguiente manera
lag_plot(m_ipc)
pyplot.show()
# Si se hace una imagen con las diferencias logarítmicas del Cierre Ajustado
# queda de la siguiente manera
lag_plot(dipc_adj)

# En esta grafica se puede ver que no existe una correlacion importante
# o significativa visualmente, la mayoria de los datos se centran
# una grafica con una tendencia o cuyos datos se distribuyen hacia alguna
# direccion mostraria una correlacion mas marcada.

# Para poder determinar de manera más precisa la correlacion entre los datos
# se puede hacer una prueba de Pearson, o coeficiente de correlacion, en donde
# se busca determinar el grado de correlacion entre las variables o datos
# Una correlacion menor a cero o negativa indica que la correlacion es negativa
# una correlacion mayor a 0 o positiva indica que los valores estan 
# correlacionados positivamente.
# una correlacion, ya sea negativa o positiva, pero mayor a +-0.5
# es indicio de una alta correlacion.

from pandas import DataFrame
from pandas import concat

ipc_corr = concat([ipc['Return'].shift(1), ipc['Return']], axis = 1)
ipc_corr.columns = ['t-1','t+1']
resultado = ipc_corr.corr()
print(resultado)

m_ipc_corr = concat([m_ipc.shift(1), m_ipc], axis = 1)
m_ipc_corr.columns = ['t-1','t+1']
m_ipc_corr_resultado = m_ipc_corr.corr()
print(m_ipc_corr_resultado)

# Como podemos ver, la correlacion entre un dato t+1 y un dato t-1 es
# mayor a 0, por lo tanto positiva, y menor a 0.5, por lo tanto
# no existe una correlacion significativa pero si positiva entre los datos.

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(ipc['Return'])
pyplot.show()

autocorrelation_plot(m_ipc)

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(ipc['Return'], lags = 31)
pyplot.show()

plot_acf(m_ipc, lag = 31)

##################
# Filtros Kalman #
##################


# Preparacion de datos que se obtienen de la serie de tiempo

#mu = np.mean(ipc['Return'])     # media de dipc_adj
#sd = np.std(ipc['Return'])      # desviacion estanar de dipc_adj
#sigma = np.var(ipc['Return'])   # varianza de dipc_adj
#size = len(ipc['Return'])       # tamano del vector de dipc_adj
#x = ipc['Return'][0]            # primer valor del vector
#z_ipc = np.random.normal(mu, sd, size)

mu = np.mean(m_ipc)     # media de dipc_adj
sd = np.std(m_ipc)      # desviacion estanar de dipc_adj
sigma = np.var(m_ipc)   # varianza de dipc_adj
size = len(m_ipc)       # tamano del vector de dipc_adj
x = m_ipc[0]     # primer valor del vector
z_ipc = np.random.normal(mu, sd, size)

# Preparacion de datos a estimar

xhat = np.zeros(size)      # estimacion de x a posteri 
P = np.zeros(size)         # estimacion de e a posteri 
xhatminus = np.zeros(size) # estimacion de x a priori 
Pminus = np.zeros(size)    # estimacion de e a priori 
K = np.zeros(size)         # ganancia Kalman o blending factor
xreal = np.zeros(size)     # un valor real de x
z_est = np.zeros(size)     # una observacion de x

Q = sigma                  # varianza de la serie de tiempo

R = 0.1**2                 # estimacion de la varianza

# Estimaciones iniciales
xhat[0] = 0.0              # valor inicial del vector xhat
P[0] = 0.03434             # valor inicial del vector P
xreal[0]=x                 # valor inicial del vector xreal
a=1.1                      # se asigna un valor inicial a

# Filtro Kalman Clasico
for k in range(1,size):
    # time update
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(z_ipc[k]-xhatminus[k])
    P[k] = (1-K[k])*Pminus[k]
    
# Grafico de Estimaciones Iniciales
plt.figure()
plt.plot(z_ipc,'k+',label='estimacion ruido')
plt.plot(xhat,'b-',label='estimacion a posteri')
plt.axhline(x,color='g',label='valor inicial')
plt.legend()
plt.title('Estimacion vs. iteracion', fontweight='bold')
plt.xlabel('Iteracion')
plt.ylabel('Sigma')

# Grafico de Estimaciones sobre la medida
plt.figure()
valid_iter = range(1,size) # Pminus not valid at step 0
plt.plot(valid_iter,Pminus[valid_iter],label='Estimación del error a priori')
plt.title('Estimación del error a priori vs. Paso de iteración', fontweight='bold')
plt.xlabel('Iteración')
plt.ylabel('$(sigma)^2$')
plt.setp(plt.gca(),'ylim',[0,.01])
plt.show()

# Filtro Kalman Vadim
Pmax = max(P)

xhat[0] = 0.0
P[0] = 0.3434
xreal[0]=x
a=1.1

for k in range(1,size):
    # Real system
    xreal[k] = a*xreal[k-1]
    z_est[k] = xreal[k]+z_ipc[k]
    
    # time update
    xhatminus[k] = a*xhat[k-1]
    Pminus[k] = a**2*P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(z_ipc[k]-xhatminus[k])
    P[k]=Pmax

plt.plot(z_ipc,'k+',label='medición del ruido blanco')
plt.plot(xhat,'b-',label='estimación a posteriori')
#plt.plot(xreal,'g-',label='real system')
plt.legend()
plt.title('Estimación vs. Paso de iteración', fontweight='bold')
plt.xlabel('Iteración')
plt.ylabel('sigma')
