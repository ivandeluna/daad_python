# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:24:37 2017

@author: Ivandla

Ejercicios VaR con Student-t

http://www.quantatrisk.com/
2015/12/02/
student-t-distributed-linear-value-at-risk/
"""

import numpy as np
import math
from scipy.stats import skew, kurtosis, kurtosistest
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import pandas_datareader.data as web

# Fetching Yahoo! Finance from IPC data
data = web.DataReader("CX", data_source='yahoo',
                      start='2007-01-01', end='2017-09-30')['Adj Close']
px_last = np.array(data.values)
ret = px_last[1:]/px_last[:-1]-1

# Plotting IBM prince- and return-series
plt.figure(num=2, figsize=(9,6))
plt.subplot(2,1,1)
plt.plot(px_last)
plt.axis("tight")
plt.ylabel("CX Adjusted Close [USD]")
plt.subplot(2, 1, 2)
plt.plot(ret, color=(.6,.6,.6))
plt.axis("tight")
plt.ylabel("Daily Returns")
plt.xlabel("Time Period 2007-01-01 to 2017-09-30 [days]")


# Check the series for kurtosis and skewness

print("Skewness = %.2f" % skew(ret))
print("Kurtosis = %.2f" % kurtosis(ret, fisher=False))

# H_0: the null hypthesis that the kurtosis of the population from which the
#  sample was drawn is that of the normal distribution kurtosis = 3(n-1)/(n+1)
_, pvalue = kurtosistest(ret)
beta = 0.05
print("p-value = %.2f" % pvalue)
if(pvalue < beta):
    print("Reject H_0 in favour of H_1 at %.5f level\n" % beta)
else:
    print("Accept H_0 at %.5f level\n" % beta)
    
# se rechaza la hipotesis nula de que la kurtosis es de una dist normal
# por lo cual se acepta la H_1 de que no es una dist normal
    
# N(x; mu, sig) best fit (finding: mu, stdev)
mu_norm, sig_norm = norm.fit(ret)
dx = 0.0001  # resolution
x = np.arange(-0.1, 0.1, dx)
pdf = norm.pdf(x, mu_norm, sig_norm)
print("Integral norm.pdf(x; mu_norm, sig_norm) dx = %.2f" % (np.sum(pdf*dx)))
print("Sample mean  = %.5f" % mu_norm)
print("Sample stdev = %.5f" % sig_norm)
print()

# Student t best fit (finding: nu)
parm = t.fit(ret)
nu, mu_t, sig_t = parm
pdf2 = t.pdf(x, nu, mu_t, sig_t)
print("Integral t.pdf(x; mu, sig) dx = %.2f" % (np.sum(pdf2*dx)))
print("nu = %.2f" % nu)
print()

# Compute VaR
h = 1  # days
alpha = 0.01  # significance level
StudenthVaR = (h*(nu-2)/nu)**0.5 * t.ppf(1-alpha, nu)*sig_norm - h*mu_norm
NormalhVaR = norm.ppf(1-alpha)*sig_norm - mu_norm 
lev = 100*(1-alpha)
print("%g%% %g-day Student t VaR = %.2f%%" % (lev, h, StudenthVaR*100))
print("%g%% %g-day Normal VaR    = %.2f%%" % (lev, h, NormalhVaR*100))

plt.figure(num=1, figsize=(11, 6))
grey = .77, .77, .77
# main figure
plt.hist(ret, bins=50, normed=True, color=grey, edgecolor='none')
plt.axis("tight")
plt.plot(x, pdf, 'b', label="Normal PDF fit")
plt.axis("tight")
plt.plot(x, pdf2, 'g', label="Student t PDF fit")
plt.xlim([-0.2, 0.1])
plt.ylim([0, 50])
plt.legend(loc="best")
plt.xlabel("Daily Returns of CX")
plt.ylabel("Normalised Return Distribution")
# inset
a = plt.axes([.22, .35, .3, .4])
plt.hist(ret, bins=50, normed=True, color=grey, edgecolor='none')

plt.plot(x, pdf, 'b')

plt.plot(x, pdf2, 'g')

# Student VaR line
plt.plot([-StudenthVaR, -StudenthVaR], [0, 3], c='g')
# Normal VaR line
plt.plot([-NormalhVaR, -NormalhVaR], [0, 4], c='b')
plt.text(-NormalhVaR-0.01, 4.1, "Norm VaR", color='b')
plt.text(-StudenthVaR-0.0171, 3.1, "Student t VaR", color='g')
plt.xlim([-0.07, -0.02])
plt.ylim([0, 5])
plt.show()

















