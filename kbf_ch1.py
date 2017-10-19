<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:54:05 2017

@author: Ivandla

"""

"""
Kalman and Bayesian Filters in Python

Chapter 1

g-h Filter
"""

import matplotlib.pyplot as plt

''' ejercicio 1 barras de error'''
plt.errorbar([160], [1], xerr=3, fmt='o',label='A', capthick=2, capsize=10)
plt.errorbar([170], [1.05], xerr=3*3, fmt='o', label='B', capthick=2, capsize=10)
plt.ylim(0, 2)
plt.xlim(145, 185)
plt.legend()
plt.gca().axes.yaxis.set_ticks([])
plt.show()

''' ejercicio 2 '''
#N = 10000
#import random
#weight_sum = 0
#for i in range(N):
#    #choose a random number between 160 and 170, assuming true weight of 165
#    measurement = random.uniform(160,170)
#    weight_sum += measurement
#    
#average = weight_sum / N
#print('Average of measurements is {:.4f}'.format(average))

''' ejercicio 3 '''
#plt.errorbar([1,2,3],[170,161,169],
#             xerr=0, yerr=10, fmt='o', capthick=2, capsize=10)
#plt.plot([1,3], [180,160], c='g', ls='--')
#plt.plot([1,3], [170,170], c='g', ls='--')
#plt.plot([1,3], [160,175], c='g', ls='--')
#plt.plot([1,2,3], [180,152, 179], c='g', ls='--')
#plt.xlim(0,4);plt.ylim(150,185)
#plt.xlabel('day')
#plt.ylabel("lbs")
#plt.show()

''' ejercicio 4 '''
#
#plt.errorbar(range(1,11),[169,170,169,171,170,171,169,170,169,170],
#             xerr=0,yerr=6, fmt='o', capthick=2, capsize=10)
#plt.plot([1,10],[169,170.5], c='g', ls='--')
#plt.xlim(0,11); plt.ylim(150,185)
#
#plt.xlabel('day')
#plt.ylabel('lbs')
#plt.show()

''' ejercicio 5 '''

#weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
#           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]
#
#plt.errorbar(range(1,13), weights,
#             xerr=0, yerr=6, fmt='o', capthick=2, capsize=10)
#
#plt.xlim(0,13); plt.ylim(145, 185)
#plt.xlabel('day')
#plt.ylabel('weight (lbs)')
#plt.show()
































=======
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:54:05 2017

@author: Ivandla

"""

"""
Kalman and Bayesian Filters in Python

Chapter 1

g-h Filter
"""

import matplotlib.pyplot as plt

''' ejercicio 1 barras de error'''
plt.errorbar([160], [1], xerr=3, fmt='o',label='A', capthick=2, capsize=10)
plt.errorbar([170], [1.05], xerr=3*3, fmt='o', label='B', capthick=2, capsize=10)
plt.ylim(0, 2)
plt.xlim(145, 185)
plt.legend()
plt.gca().axes.yaxis.set_ticks([])
plt.show()

''' ejercicio 2 '''
#N = 10000
#import random
#weight_sum = 0
#for i in range(N):
#    #choose a random number between 160 and 170, assuming true weight of 165
#    measurement = random.uniform(160,170)
#    weight_sum += measurement
#    
#average = weight_sum / N
#print('Average of measurements is {:.4f}'.format(average))

''' ejercicio 3 '''
#plt.errorbar([1,2,3],[170,161,169],
#             xerr=0, yerr=10, fmt='o', capthick=2, capsize=10)
#plt.plot([1,3], [180,160], c='g', ls='--')
#plt.plot([1,3], [170,170], c='g', ls='--')
#plt.plot([1,3], [160,175], c='g', ls='--')
#plt.plot([1,2,3], [180,152, 179], c='g', ls='--')
#plt.xlim(0,4);plt.ylim(150,185)
#plt.xlabel('day')
#plt.ylabel("lbs")
#plt.show()

''' ejercicio 4 '''
#
#plt.errorbar(range(1,11),[169,170,169,171,170,171,169,170,169,170],
#             xerr=0,yerr=6, fmt='o', capthick=2, capsize=10)
#plt.plot([1,10],[169,170.5], c='g', ls='--')
#plt.xlim(0,11); plt.ylim(150,185)
#
#plt.xlabel('day')
#plt.ylabel('lbs')
#plt.show()

''' ejercicio 5 '''

#weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
#           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]
#
#plt.errorbar(range(1,13), weights,
#             xerr=0, yerr=6, fmt='o', capthick=2, capsize=10)
#
#plt.xlim(0,13); plt.ylim(145, 185)
#plt.xlabel('day')
#plt.ylabel('weight (lbs)')
#plt.show()
































>>>>>>> 423a703f2b7a2c2c26ecbffb61fde230b35afe9a
