<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 23:52:55 2017

@author: Ivandla

Scrip para aplicar Filtros Kalman al análisis histórico del IPC de la BMV.

Periodo: Enero 2007 a Septiembre 2017

"""

# Aplicacion de KF para el IPC
# Utilizando datos de Yahoo Finance
# desde Enero de 2007 hasta Septiembre del 2017

######################################
# Importacion de Datos usando Pandas #
######################################

import pandas_datareader.data as dr  # Lector de datos
import datetime # Manejo de fechas
import matplotlib.pyplot as plt # Graficador
import numpy as np
%matplotlib inline

# Se crean dos variables, start y end
# en donde se especificara el rango de tiempo para
# poder hacer un query a la base de datos (BD)

start = datetime.datetime(2017,1,1)   # Fecha Inicio
end = datetime.datetime(2017,9,30)    # Ultima fecha

# Para poder bajar los datos de Yahoo Finance
# es necesario utilizar el ticker que utiliza la BD de Yahoo
# en este caso para el IPC, el ticker es ^MXX pero
# su en la URL se establece como %5EMXX

# Variable IPC
ipc = dr.DataReader("%5EMXX", 'yahoo', start, end) 

# Al hacer uso de Yahoo Finance para descargar datos
# se obtiene un data frame con diferentes columnas
# Open, High, Low, Close, Adjusted Close, Volume
# Para terminos de analisis de datos vamos a usar
# la variable de Adjusted Close o Cierre ajustado
# ya que integra dentro de los datos la
# informacion de movimientos corporativos como
# splits, dividendos o derechos.

##########################################################
# Analisis preliminar de los datos y seleccion de series #
##########################################################

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

plt.plot(ipc.iloc[1:2700, 4], label='IPC Cierre Ajustado') 


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
ipc_adj = ipc.iloc[1:2700,4]
dipc_adj = dipc.iloc[1:2700,4]

# Grafico de ipc_adj
plt.plot(ipc_adj, label = 'IPC Cierre Ajustado')
plt.plot(dipc_adj, label = 'IPC Cierre Ajustado Dif Logaritmicas')

dipc.plot()

# A diferencia del tratamiento de datos como vectores, se puede
# aprovechar el df, de la siguiente maneta

# se crea una funcino de diferencias logaritmicas
ipc['Return'] = np.log(ipc['Close']/ipc['Close'].shift(1))

# se hace una grafica del precio y las diferencias (rendimientos)
ipc[['Close', 'Return']].plot(subplots=True, style='b', figsize=(8, 5))

###################################
# Estimacion de un modelo AR(a,r) #
###################################

# Para poder crear un modelo AR(p,q) es necesario conocer la naturaleza
# de la serie con la que estamos tratando. En especifico ver las relaciones
# entre los datos o su correlacion.

# Una manera visual de hacer esto es con graficas

from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot

lag_plot(ipc['Return'])
pyplot.show()

# En esta grafica se puede ver que no existe una correlacion importante
# o significativa visualmente, la mayoria de los datos se centran
# una grafica con una tendencia o cuyos datos se distribuyen hacia alguna
# direccion mostraria una correlacion mas marcada.

# Para poder determinar de manera mas precisa la correlacion entre los datos
# se puede hacer una prueba de Pearson, o coeficiente de correlacion, en donde
# se busca determinar el grado de correlacion entre las variables o datos
# Una correlacion menor a cero o negativa indica que la correlacion es negativa
# una correlacion mayor a 0 o positiva indica que los valores estan 
# correlacionados positivamente.
# una correlacion, ya sea negativa o positiva, pero mayor a +-0.5
# es indicio de una alta correlacion.

from pandas import DataFrame
from pandas import concat

df_corr = concat([ipc['Return'].shift(1), ipc['Return']], axis = 1)
df_corr.columns = ['t-1','t+1']
resultado = df_corr.corr()
print(resultado)

# Como podemos ver, la correlacion entre un dato t+1 y un dato t-1 es
# mayor a 0, por lo tanto positiva, y menor a 0.5, por lo tanto
# no existe una correlacion significativa pero si positiva entre los datos.

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(ipc['Return'])
pyplot.show()

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(ipc['Return'], lags=31)
pyplot.show()

##################
# Filtros Kalman #
##################


# Preparacion de datos que se obtienen de la serie de tiempo

mu = np.mean(ipc['Return'])     # media de dipc_adj
sd = np.std(ipc['Return'])      # desviacion estanar de dipc_adj
sigma = np.var(ipc['Return'])   # varianza de dipc_adj
size = len(ipc['Return'])       # tamano del vector de dipc_adj
x = ipc['Return'][0]            # primer valor del vector
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
P[0] = 0.03434                # valor inicial del vector P
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
plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
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

plt.plot(z_ipc,'k+',label='noisy measurements')
plt.plot(xhat,'b-',label='a posteri estimate')
#plt.plot(xreal,'g-',label='real system')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('sigma')
=======
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 23:52:55 2017

@author: Ivandla

Scrip para aplicar Filtros Kalman al análisis histórico del IPC de la BMV.

Periodo: Enero 2007 a Septiembre 2017

"""

# Aplicacion de KF para el IPC
# Utilizando datos de Yahoo Finance
# desde Enero de 2007 hasta Septiembre del 2017

######################################
# Importacion de Datos usando Pandas #
######################################

import pandas_datareader.data as dr  # Lector de datos
import datetime # Manejo de fechas
import matplotlib.pyplot as plt # Graficador
import numpy as np
%matplotlib inline

# Se crean dos variables, start y end
# en donde se especificara el rango de tiempo para
# poder hacer un query a la base de datos (BD)

start = datetime.datetime(2007,1,1)   # Fecha Inicio
end = datetime.datetime(2017,9,30)    # Ultima fecha

# Para poder bajar los datos de Yahoo Finance
# es necesario utilizar el ticker que utiliza la BD de Yahoo
# en este caso para el IPC, el ticker es ^MXX pero
# su en la URL se establece como %5EMXX

# Variable IPC
ipc = dr.DataReader("%5EMXX", 'yahoo', start, end) 

# Al hacer uso de Yahoo Finance para descargar datos
# se obtiene un data frame con diferentes columnas
# Open, High, Low, Close, Adjusted Close, Volume
# Para terminos de analisis de datos vamos a usar
# la variable de Adjusted Close o Cierre ajustado
# ya que integra dentro de los datos la
# informacion de movimientos corporativos como
# splits, dividendos o derechos.

##########################################################
# Analisis preliminar de los datos y seleccion de series #
##########################################################

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

plt.plot(ipc.iloc[1:2700, 4], label='IPC Cierre Ajustado') 


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
ipc_adj = ipc.iloc[1:2700,4]
dipc_adj = dipc.iloc[1:2700,4]

# Grafico de ipc_adj
plt.plot(ipc_adj, label = 'IPC Cierre Ajustado')
plt.plot(dipc_adj, label = 'IPC Cierre Ajustado Dif Logaritmicas')

dipc.plot()

#

ipc['Return'] = np.log(ipc['Close']/ipc['Close'].shift(1))

ipc[['Close', 'Return']].plot(subplots=True, style='b', figsize=(8, 5))
###################################
# Estimacion de un modelo AR(a,r) #
###################################





##################
# Filtros Kalman #
##################


# Preparacion de datos que se obtienen de la serie de tiempo

mu = np.mean(dipc_adj)     # media de dipc_adj
sd = np.std(dipc_adj)      # desviacion estanar de dipc_adj
sigma = np.var(dipc_adj)   # varianza de dipc_adj
size = len(dipc_adj)       # tamano del vector de dipc_adj
x = dipc_adj[0]            # primer valor del vector
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
P[0] = 1.0                 # valor inicial del vector P
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
plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('$(sigma)^2$')
plt.setp(plt.gca(),'ylim',[0,.01])
plt.show()

# Filtro Kalman Vadim
Pmax = max(P)

xhat[0] = 0.0
P[0] = 1.0
xreal[0]=x
a=1.1

for k in range(1,size):
    # Real system
    xreal[k] = a*xreal[k-1]
    z_est[k] = xreal[k]+z[k]
    
    # time update
    xhatminus[k] = a*xhat[k-1]
    Pminus[k] = a**2*P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
    P[k]=Pmax

plt.plot(z,'k-',label='noisy measurements')
plt.plot(xhat,'b-',label='a posteri estimate')
plt.plot(xreal,'g-',label='real system')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('sigma')
>>>>>>> 423a703f2b7a2c2c26ecbffb61fde230b35afe9a
