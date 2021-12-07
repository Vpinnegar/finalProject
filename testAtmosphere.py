# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:20:50 2021

@author: starv
"""

#Cloud SImulation 

#produce a cloud in 2 dimensions plus magnitude


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
import simpy
import random
import statistics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from scipy import interpolate


''' functions '''
def CollisionEff(y,r1,r2):
    E = y**2/(r1+r2)**2
    return E

""" Relative Sizes of droplets
CCN r = 0.1 nm, n = 10**6 #/liter, v = 0.0001 cm / s
Cloud Droplet r = 10 nm n = 10**6 v = 1
Large CD r = 50, n = 10**3, v = 27 
borderline r = 100 v = 70,
raindrop r = 1000 n = 1 v = 650 """

def Updraft(Dropvel, upvel):
    Final = - int(Dropvel) + upvel
    return Final

# Altitude = np.linspace(0,10000,2000)
# NumofParticles = np.linspace(0,2000,2000)
# Horizontal = np.linspace(-10,10,100)


# class CloudMicrophysics(object):
#     def __init__(self, env, num_particles):
#         self.env = env
#         self.particles = simpy.Resource(env, num_particles)
    
#     def radius(self, particle):
#         yield self.env.rad(random.uniform(0.1,1))
def RandomBackscatter(Alt, points, mid, varin):      
    height = np.linspace(0, Alt, num=points)
    
    # 1440 per day
    time = np.linspace(0, 24, 1440)
    
    atmos = np.zeros((25, points))
    
    std = np.random.normal(200, 50, 1)
    mean = np.random.normal(2000, 200, 1)
    scale = np.random.normal(mid, varin, 1)
    
    for i in range(len(atmos)):
    	std += np.random.normal(0, 50, 1)
    	while std < 150:
    		std += np.random.normal(0, 50, 1)
    	mean += np.random.normal(0, 200, 1)
    	while mean < 800:
    		mean += np.random.normal(0, 200, 1)
    	scale += np.random.normal(0, varin, 1)
    	atmos[i] = 100 + scale * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(height-mean)**2/(2*std**2))
    
    # for i in range(len(atmos)):
    # 	std = np.random.normal(300, 3, 1)
    # 	mean = np.random.normal(6000, 200, 1)
    # 	scale = np.random.normal(100, 50, 1)
    # 	atmos[i] += scale * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(height-mean)**2/(2*std**2))
    
    f = interpolate.interp2d(range(0, 25, 1), height, np.transpose(atmos), kind='linear')
    
    atmos_full = f(time, height)
    
    df = pd.DataFrame(data=atmos_full)
    
    return atmos_full
sns.set_style('darkgrid', {'xtick.bottom': True, 'ytick.left': True, 'axes.edgecolor': '0.15'})
sns.set_context({'font.size': '14'})
df = RandomBackscatter(10000, 2000, 1e-6, 1e-10)

fig = plt.figure(figsize=(17.12, 9.6))
g = sns.heatmap(df, cmap='jet')
plt.gca().invert_yaxis()
g.set_xticks(np.linspace(0,1440,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
g.set_yticks([0, 500, 1000, 1500, 2000])
g.set_yticklabels([0, 2500, 5000, 7500, 10000])
        
df2 = RandomBackscatter(10000, 2000, 1e-1, 1e-5)
fig = plt.figure(figsize=(17.12, 9.6))
g = sns.heatmap(df2, cmap='jet')
plt.gca().invert_yaxis()
g.set_xticks(np.linspace(0,1440,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
g.set_yticks([0, 500, 1000, 1500, 2000])
g.set_yticklabels([0, 2500, 5000, 7500, 10000])


fig = plt.figure(figsize=(17.12, 9.6))
g = sns.heatmap(abs(df/df2), cmap='jet', vmin = 1e-13, vmax = 1e-2, norm=LogNorm())
plt.gca().invert_yaxis()
g.set_xticks(np.linspace(0,1440,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
g.set_yticks([0, 500, 1000, 1500, 2000])
g.set_yticklabels([0, 2500, 5000, 7500, 10000])
