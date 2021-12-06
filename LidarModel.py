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
import miepython
from testAtmosphere import RandomBackscatter
import seaborn as sns
""" Constants """
c  = 299702547
overlap = np.ones(6000)
km1 = np.linspace(0,1,200)
for i in range(len(km1)):
    overlap[i] = km1[i]
## Scattering efficiencies for Cloud, Aerosol, Thick cloud/drizzle
aerosolindex = 1.5
"""BEta"""
Beta = np.linspace(0,2000,2000)
for i in range(len(Beta)):
    Beta[i] = np.random.random()*3
    



""" Functions """
sealevel = Atmosphere(0)
densitysea = sealevel.density
##Ambiance -- ICAO standard atmosphere 1993
allheights = Atmosphere(np.linspace(0,10000,2000))
def scatlimit(r,wavelength):
    return 2*np.pi*r/wavelength
def waterDroprefrac(lambda0):
    m2 = 1.0
    m2 += 5.666959820E-1 / (1.0 - 5.084151894E-3 / lambda0**2)
    m2 += 1.731900098E-1 / (1.0 - 1.818488474E-2 / lambda0**2)
    m2 += 2.095951857E-2 / (1.0 - 2.625439472E-2 / lambda0**2)
    m2 += 1.125228406E-1 / (1.0 - 1.073842352E1 / lambda0**2)
    m = np.sqrt(m2)
    return m

## Pyatmos COesa standard atmosphere
coesa76_geom = coesa76(np.linspace(0,10000,2000))
def VEGABetaMol(wave, angle, bins, binsize, maxheight, n, r):
    TzPz = coesa76(np.linspace(0,maxheight,bins)).P/coesa76(np.linspace(0,maxheight,bins)).T
    betamol = np.zeros(len(TzPz))
    # V = np.zeros(len(TzPz))
    K = 1.38e-23
    for i in range(len(TzPz)):
        # print(TzPz[i])
        # V[i] = (np.pi/3)*((((i+1)*binsize)**3)* np.tan(angle) - (((i)*binsize)**3)* np.tan(angle))
        betamol[i] = (TzPz[i]/K) *(16*(np.pi**4))*(((n**2 + 2)/(n**2 - 1))**2)* r**6 * wave**(-4)
    return betamol
# BetaMolVEGAICAO = VEGABetaMol(allheights.pressure/allheights.temperature, float(5.32*10**(-7))) ##Ambiance
# BetaMolVEGACOESA = VEGABetaMol(coesa76(np.linspace(0,10,2000)).P/coesa76(np.linspace(0,10,2000)).T, float(5.32*10**(-7)))

# BetaMolVEGAICAORadar = VEGABetaMol(allheights.pressure/allheights.temperature, float(0.0086)) ##Ambiance
# BetaMolVEGACOESARadar = VEGABetaMol(coesa76(np.linspace(0,10,2000)).P/coesa76(np.linspace(0,10,2000)).T, float(0.0086))
def MieBackscatter(time,height, inpu, particler, m, lambd, binsize, angle):
    back = np.zeros((height, time))
    # V = np.zeros(height)
    x = 2*np.pi*particler/lambd
    K = 1.38e-23
    Qext, qsca, Qback, g = miepython.mie(m,x)
    print(Qback)
    for sec in range(time):
        for al in range(height):
            # print(sec,al)
            # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
            back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
            
            
    return back
def RadBackscatter(time,height, inpu, r, n, wave, binsize, angle):
    back = np.zeros((height, time))
    # V = np.zeros(height)
    # x = 2*np.pi*particler/lambd
    K = 1.38e-23
    # Qext, qsca, Qback, g = miepython.mie(n,x)
    for sec in range(time):
        for al in range(height):
            # print(sec,al)
            # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
            back[al,sec] = (inpu[al,sec]/K) *(16*(np.pi**4))*(((n**2 + 2)/(n**2 - 1))**2)* r**6 * wave**(-4)
            
            
    return back
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
def ReflectivitytoBackscatter(Z, m , wavelength):
    for i in range(len(Z)):
        beta = ((m**2 -1)/(m**2 + 2))*(np.pi**4/(4*wavelength**4))*Z[i]
    return beta

# def BackscattertoReflectivity(B, m , wavelength):
#     for i in range(len(Z)):
#         beta = ((m**2 -1)/(m**2 + 2))*(np.pi**4/(4*wavelength**4))*Z[i]
#     return beta

def Radar(Power, binlength, Gain, theta, phi, K, loss, Beta,\
 wavelength, Range, m):
    Z_e = Beta * (m**2-1)/(m**2+2) * 4 * wavelength**4 / np.pi**4
    P = np.pi**3 * Power * Gain**2 * theta*phi * binlength * K**2 * loss * Z_e
    return P
df = RandomBackscatter(10000,2000) 

# molback = VEGABetaMol(float(5.32*10**(-7)), 0.00005, 2000, 5, 10, 1.0003, 1e-9)

# molbackrad = VEGABetaMol(0.0086, 0.002617, 2000, 5, 10, 1.0003, 1e-9)

partbackcloudLi = MieBackscatter(1440,2000, df, 10e-6, waterDroprefrac(5.32*10**(-7)), 5.32*10**(-7), 5, 0.00005)

partbackcloudra = MieBackscatter(1440,2000, df, 10e-6, waterDroprefrac(0.0086), 0.0086, 5, 0.00005)


# for i in range(len(partbackcloudLi[0])):
#     partbackcloudLi[:,0] = partbackcloudLi[:,i] + molback

# for i in range(len(partbackcloudLi[0])):
#     partbackcloudLi[:,0] = partbackcloudra[:,i] + molbackrad

# Try = Lidar(2, 5, 0.4, 0.9,10000 , BetaMolVEGACOESA, 0.98)
# heights = np.linspace(0,10000,2000)
# plt.plot(Try, heights)
# # plt.xlim(0,0.00000000001)
# plt.figure()
# plt.plot(BetaMolVEGACOESA, heights)
fig = plt.figure(figsize=(17.12, 9.6))
g = sns.heatmap(df, cmap='jet')
plt.gca().invert_yaxis()
g.set_xticks(np.linspace(0,1440,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
g.set_yticks([0, 500, 1000, 1500, 2000])
g.set_yticklabels([0, 2500, 5000, 7500, 10000])









