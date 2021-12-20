import numpy as np
from scipy import interpolate
import sys

def randomAtmosphere(Alt, height_bins, time_bins, mean_of_scale, mean_of_std, std_of_std, mean_of_mean, std_of_mean, cloud_type):      
    height = np.linspace(0, Alt, num=height_bins)
    time = np.linspace(0, 24, time_bins)
    atmos = np.zeros((25, height_bins))
    std = np.random.normal(mean_of_std, std_of_std, 1)
    mean = np.random.normal(mean_of_mean, std_of_mean, 1)
    scale = mean_of_scale
    for i in range(len(atmos)):
        std += np.random.normal(std_of_std/2, std_of_std, 1)
        while std < std_of_std/3:
            std += np.random.normal(std_of_std/2, std_of_std, 1)
        mean += np.random.normal(0, std_of_mean, 1)
        while mean < 800:
            mean += np.random.normal(0, std_of_mean, 1)
        atmos[i] = 100 + scale * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(height-mean)**2/(2*std**2))
    f = interpolate.interp2d(range(0, 25, 1), height, np.transpose(atmos), kind='linear')
    atmos_full = f(time, height)

    particler = np.zeros((height_bins, time_bins))
    for sec in range(time_bins):
        sys.stdout.write('\rCalculating time bin {:<4d} of {:<4d}'.format(sec+1, time_bins))
        sys.stdout.flush() 
        for al in range(height_bins):
            if cloud_type == 'cloud':
                if atmos_full[al,sec] >= 300:
                    particler[al,sec] = np.random.normal(5e-5, 1e-5, 1)
                elif atmos_full[al,sec] >= 250:
                    particler[al,sec] = np.random.normal(5e-6, 1e-6, 1)
                elif atmos_full[al,sec] >= 200:
                    particler[al,sec] = np.random.normal(5e-7, 1e-7, 1)
                else:
                    particler[al,sec] = np.random.normal(5e-10, 1e-10, 1)
            elif cloud_type == 'aerosol':
                if atmos_full[al,sec] >= 300:
                    particler[al,sec] = np.random.normal(5e-7, 1e-7, 1)
                elif atmos_full[al,sec] >= 250:
                    particler[al,sec] = np.random.normal(5e-8, 1e-8, 1)
                elif atmos_full[al,sec] >= 200:
                    particler[al,sec] = np.random.normal(5e-9, 1e-9, 1)
                else:
                    particler[al,sec] = np.random.normal(5e-10, 1e-10, 1)
			
    print()
    return atmos_full, particler