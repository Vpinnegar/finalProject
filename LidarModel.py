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
# import netCDF4 as nc # import libraries
import numpy as np
# import sklearn
import scipy
# import glob
# import math 
# import statsmodels
# from scipy.integrate import quad
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from scipy.signal import savgol_filter
from ambiance import Atmosphere
from pyatmos import coesa76
import miepython
from matplotlib.colors import LogNorm
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
def VEGABetaMolM(wave, angle, bins, binsize, maxheight, n, r):
    TzPz = coesa76(np.linspace(0,maxheight,bins)).P/coesa76(np.linspace(0,maxheight,bins)).T
    betamol = np.zeros(len(TzPz))
    # V = np.zeros(len(TzPz))
    K = 1.38e-23
    # print((((n**2 + 2)/(n**2 - 1))**2))
    for i in range(len(TzPz)):
        
        # print(TzPz[i])
        # V[i] = (np.pi/3)*((((i+1)*binsize)**3)* np.tan(angle) - (((i)*binsize)**3)* np.tan(angle))
        betamol[i] = (TzPz[i]**-1 * K) *(16*(np.pi**4))*(((n**2 + 2)/(n**2 - 1))**2)* r**6 * wave**(-4)
    return betamol
## Pyatmos COesa standard atmosphere
coesa76_geom = coesa76(np.linspace(0,10000,2000))
def VEGABetaMol(time,height,inpu, wave, angle, bins, binsize, maxheight, n, r):
    # TzPz = coesa76(np.linspace(0,maxheight,bins)).P/coesa76(np.linspace(0,maxheight,bins)).T
    betamol = np.zeros((height, time))
    # V = np.zeros(len(TzPz))
    K = 1.38e-23
    for i in range(time):
        for j in range(height):
        # print(TzPz[i])
        # V[i] = (np.pi/3)*((((i+1)*binsize)**3)* np.tan(angle) - (((i)*binsize)**3)* np.tan(angle))
            betamol[j,i] = (inpu[j,i] /K) *5.31e-22* wave**(-4)
    return betamol
# BetaMolVEGAICAO = VEGABetaMol(allheights.pressure/allheights.temperature, float(5.32*10**(-7))) ##Ambiance
# BetaMolVEGACOESA = VEGABetaMol(coesa76(np.linspace(0,10,2000)).P/coesa76(np.linspace(0,10,2000)).T, float(5.32*10**(-7)))

# BetaMolVEGAICAORadar = VEGABetaMol(allheights.pressure/allheights.temperature, float(0.0086)) ##Ambiance
# BetaMolVEGACOESARadar = VEGABetaMol(coesa76(np.linspace(0,10,2000)).P/coesa76(np.linspace(0,10,2000)).T, float(0.0086))
def MieBackscatter(time,height, inpu, particler, m, lambd, binsize, angle, typ):
    back = np.zeros((height, time))
    # V = np.zeros(height)
    
    K = 1.38e-23
    
    # print(Qback)
    if typ == 'cloud':
        
        for sec in range(time):
            # print(sec)
            for al in range(height):
                # print(sec,al)
                if inpu[al,sec] >= 300:
                    
                    particler = np.random.normal(50e-9, 10e-9, 1) * inpu[al,sec]
                    x = 2*np.pi*particler/lambd
                    Qext, qsca, Qback, g = miepython.mie(m,x)
                # print(sec,al)
                # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                elif 299 > inpu[al,sec] >= 250: 
                    particler = np.random.normal(50e-10, 25e-10, 1) * inpu[al,sec]
                    
                    x = 2*np.pi*particler/lambd
                    Qext, qsca, Qback, g = miepython.mie(m,x)
                # print(sec,al)
                # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                elif 250 > inpu[al,sec] >= 200: 
                    particler = np.random.normal(90e-10, 55e-10, 1) * inpu[al,sec]
                    
                    x = 2*np.pi*particler/lambd
                    Qext, qsca, Qback, g = miepython.mie(m,x)
                # print(sec,al)
                # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                    
                else: 
                    particler = np.random.normal(50e-11, 25e-11, 1) * inpu[al,sec]
                    
                    x = 2*np.pi*particler/lambd
                    Qext, qsca, Qback, g = miepython.mie(m,x)
                # print(sec,al)
                # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
            
    if typ == 'aero':
        
        for sec in range(time):
            # print(sec)
            for al in range(height):
                # print(sec,al)
                if inpu[al,sec] >= 300:
                    
                    particler = np.random.normal(50e-10, 10e-10, 1) * inpu[al,sec]
                    x = 2*np.pi*particler/lambd
                    Qext, qsca, Qback, g = miepython.mie(m,x)
                # print(sec,al)
                # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                elif 299 > inpu[al,sec] >= 250: 
                    particler = np.random.normal(50e-10, 25e-10, 1) * inpu[al,sec]
                    
                    x = 2*np.pi*particler/lambd
                    Qext, qsca, Qback, g = miepython.mie(m,x)
                # print(sec,al)
                # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                elif 250 > inpu[al,sec] >= 200: 
                    particler = np.random.normal(90e-10, 55e-10, 1) * inpu[al,sec]
                    
                    x = 2*np.pi*particler/lambd
                    Qext, qsca, Qback, g = miepython.mie(m,x)
                # print(sec,al)
                # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                    
                else: 
                    particler = np.random.normal(50e-11, 25e-11, 1) * inpu[al,sec]
                    
                    x = 2*np.pi*particler/lambd
                    Qext, qsca, Qback, g = miepython.mie(m,x)
                # print(sec,al)
                # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback       
    return back
def RadBackscatter(time,height, inpu, n, wave, binsize, angle, typ):
    back = np.zeros((height, time))
    # V = np.zeros(height)
    # x = 2*np.pi*particler/lambd
    K = 1.38e-23
    bigboy = np.abs(((n**2 + 2)/(n**2 - 1)))**2
    # Qext, qsca, Qback, g = miepython.mie(n,x)
    if typ == 'cloud':
        
        for sec in range(time):
            print(sec)
            for al in range(height):
                if inpu[al,sec] >= 300:
                    r = np.random.normal(50e-8, 10e-8, 1) * inpu[al,sec]
                        # print(sec,al)
                        # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                elif 299 > inpu[al,sec] >= 250: 
                        r = np.random.normal(90e-10, 55e-10, 1) * inpu[al,sec]
                        
                        
                    # print(sec,al)
                    # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                        back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                elif 250 > inpu[al,sec] >= 200: 
                        r = np.random.normal(90e-10, 55e-10, 1) * inpu[al,sec]
                        
                        
                    # print(sec,al)
                    # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                        back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                else: 
                    r = np.random.normal(50e-11, 25e-11, 1) * inpu[al,sec]
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
        else:
            for sec in range(time):
                print(sec)
                for al in range(height):
                    if inpu[al,sec] >= 300:
                        r = np.random.normal(50e-10, 10e-9, 1) * inpu[al,sec]
                            # print(sec,al)
                            # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                        back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                    elif 299 > inpu[al,sec] >= 250: 
                            r = np.random.normal(50e-10, 25e-10, 1) * inpu[al,sec]
                            
                            
                        # print(sec,al)
                        # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                            back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                    elif 250 > inpu[al,sec] >= 200: 
                            r = np.random.normal(90e-10, 55e-10, 1) * inpu[al,sec]
                            
                            
                        # print(sec,al)
                        # V[al] = (np.pi/3)*((((al+1)*binsize)**3)* np.tan(angle) - (((al)*binsize)**3)* np.tan(angle))
                            back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                    else: 
                        r = np.random.normal(50e-11, 25e-11, 1) * inpu[al,sec]
                        back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
            
                
                
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
df = RandomBackscatter(10000, 2000, 4e5, 50)
df2 = RandomBackscatter(10000, 2000, 4e4, 5)

molback = VEGABetaMolM(float(5.32*10**(-7)), 0.00005, 2000, 5, 10, 1.0003, 1e-9)

molbackrad = VEGABetaMolM(0.0086, 0.002617, 2000, 5, 10, 1.0003, 1e-9)

partbackcloudLi = MieBackscatter(1440,2000, df, 10e-6, 1.33, 5.32*10**(-7), 5, 0.00005, 'aero')

partbackcloudUGHHH = MieBackscatter(1440,2000, df, 10e-6, 5 + 2.5j,0.0086, 5, 0.00005, 'aero')

partbackcloudra = RadBackscatter(1440,2000, df,5 + 2.5j, 0.0086, 5, 1, 'aero')

# plt.plot

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
plt.title("Lidar")
g = sns.heatmap(partbackcloudLi, cmap='jet',  norm = LogNorm(), vmin = 1e14, vmax = 1e19)
plt.gca().invert_yaxis()
g.set_xticks(np.linspace(0,1440,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
g.set_yticks([0, 500, 1000, 1500, 2000])
g.set_yticklabels([0, 2500, 5000, 7500, 10000])

fig2 = plt.figure(figsize=(17.12, 9.6))
plt.title("Radar Mie")
g = sns.heatmap(partbackcloudUGHHH, cmap='jet',  norm = LogNorm(),vmin = 1e7, vmax = 1e14)
plt.gca().invert_yaxis()
g.set_xticks(np.linspace(0,1440,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
g.set_yticks([0, 500, 1000, 1500, 2000])
g.set_yticklabels([0, 2500, 5000, 7500, 10000])
        

fig3 = plt.figure(figsize=(17.12, 9.6))
plt.title("Radar")
g = sns.heatmap(partbackcloudra, cmap='jet', vmin = 1e10, vmax = 1e14, norm = LogNorm())
plt.gca().invert_yaxis()
g.set_xticks(np.linspace(0,1440,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
g.set_yticks([0, 500, 1000, 1500, 2000])
g.set_yticklabels([0, 2500, 5000, 7500, 10000])


fig4 = plt.figure(figsize=(17.12, 9.6))
print(partbackcloudLi/partbackcloudra)
plt.title("Colour Ratio")
g = sns.heatmap(partbackcloudUGHHH/partbackcloudLi, cmap='jet',vmin = 1e-13, vmax = 1e-2,  norm = LogNorm())
plt.gca().invert_yaxis()
g.set_xticks(np.linspace(0,1440,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
g.set_yticks([0, 500, 1000, 1500, 2000])
g.set_yticklabels([0, 2500, 5000, 7500, 10000])


fig5 = plt.figure(figsize=(17.12, 9.6))
print(partbackcloudLi/partbackcloudra)
plt.title("Colour Ratio")
g = sns.heatmap(partbackcloudra/partbackcloudLi, cmap='jet', vmin = 1e-13, vmax = 1e-2, norm = LogNorm())
plt.gca().invert_yaxis()
g.set_xticks(np.linspace(0,1440,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
g.set_yticks([0, 500, 1000, 1500, 2000])
g.set_yticklabels([0, 2500, 5000, 7500, 10000])


# Backish = RandomBackscatter(10000, 2000, 1e10, 1e20)
# radar = RandomBackscatter(10000,2000, 1e8, 1e14)

# fig = plt.figure(figsize=(17.12, 9.6))
# plt.title("Lidar- random")
# g = sns.heatmap(Backish, cmap='jet',vmin = 1e14, vmax = 1e19,  norm = LogNorm())
# plt.gca().invert_yaxis()
# g.set_xticks(np.linspace(0,1440,25))
# g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
# g.set_yticks([0, 500, 1000, 1500, 2000])
# g.set_yticklabels([0, 2500, 5000, 7500, 10000])

# fig = plt.figure(figsize=(17.12, 9.6))
# plt.title("radar- random")
# g = sns.heatmap(radar, cmap='jet', vmin = 1e7, vmax = 1e14, norm = LogNorm())
# plt.gca().invert_yaxis()
# g.set_xticks(np.linspace(0,1440,25))
# g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
# g.set_yticks([0, 500, 1000, 1500, 2000])
# g.set_yticklabels([0, 2500, 5000, 7500, 10000])

# fig = plt.figure(figsize=(17.12, 9.6))
# plt.title("colourratio- random")
# g = sns.heatmap(radar/Backish, cmap='jet', vmin = 1e-13, vmax = 1e-2, norm = LogNorm())
# plt.gca().invert_yaxis()
# g.set_xticks(np.linspace(0,1440,25))
# g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
# g.set_yticks([0, 500, 1000, 1500, 2000])
# g.set_yticklabels([0, 2500, 5000, 7500, 10000])
