# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 00:48:39 2021

@author: Victoria
"""
"""Program Goals"""
# Lidar equation simulation 
# 1. Emulate Lidar Power for standard atmosphere
#   - Produce lidar simulation and utilize US standard atmosphere to 
'''Imports '''
from matplotlib import pyplot as plt # import libraries
import pandas as pd # import libraries
import netCDF4 as nc # import libraries
import numpy as np
import sklearn
import scipy
import glob
import math 
import statsmodels
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
from ambiance import Atmosphere
from pyatmos import coesa76

""" Constants """
c  = 299792458
overlap = np.ones(6000)
km1 = np.linspace(0,1,200)
for i in range(len(km1)):
    overlap[i] = km1[i]

"""BEta"""
Beta = np.linspace(0,2000,2000)
for i in range(len(Beta)):
    Beta[i] = np.random.random()*3
    



""" Functions """
sealevel = Atmosphere(0)
densitysea = sealevel.density
##Ambiance -- ICAO standard atmosphere 1993
allheights = Atmosphere(np.linspace(0,10000,2000))

## Pyatmos COesa standard atmosphere
coesa76_geom = coesa76(np.linspace(0,10000,2000))
def VEGABetaMol(Tz, Pz, wave):
    betamol = np.zeros(len(Pz))
    for i in range(len(Pz)):
        betamol[i] = (2.938*10**(-32))*(Pz[i]/Tz[i])*(1/(wave**(4.0117)))
    return betamol
BetaMolVEGAICAO = VEGABetaMol(allheights.temperature, allheights.pressure, float(5.32*10**(-7))) ##Ambiance
BetaMolVEGACOESA = VEGABetaMol(coesa76(np.linspace(0,10,2000)).T, coesa76(np.linspace(0,10,2000)).P, float(5.32*10**(-7)))

def Lidar(Power, binlength, Area, Eff, Range, Beta, Transmission):
    P = []
    numbins = Range/binlength
    perkm = numbins/ (Range/1000)
    overlap = np.ones(Range)
    km1 = np.linspace(0,1,int(perkm))
    for i in range(len(km1)):
        overlap[i] = km1[i]
    for height in range(len(Beta)):
        PH = Power * (binlength) * Area * Eff * (1/(Range**2))*Beta[height]*Transmission
        P.append(PH)
    return P

def Radar(Power, binlength, Gain, theta, phi, dielectric, loss, reflectivity,\
 wavelength, Range, m):
    Z_e = Beta * (m**2-1)/(m**2+2) * 4 * wavelength**4 / np.pi**4
    P = np.pi**3 * Power * Gain**2 * theta*phi * binlength * K**2 * l * Z_e
    return P

Try = Lidar(2, 5, 0.4, 0.9,10000 , BetaMolVEGACOESA, 0.98)
heights = np.linspace(0,10000,2000)
plt.plot(Try, heights)
# plt.xlim(0,0.00000000001)
plt.figure()
plt.plot(BetaMolVEGACOESA, heights)


        









