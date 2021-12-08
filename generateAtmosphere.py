import numpy as np
from scipy import interpolate

def RandomBackscatter(Alt, bins, mean_of_scale, std_of_scale, mean_of_std, std_of_std, mean_of_mean, std_of_mean):      
    height = np.linspace(0, Alt, num=bins)
    time = np.linspace(0, 24, 1440)
    atmos = np.zeros((25, bins))
    std = np.random.normal(mean_of_std, std_of_std, 1)
    mean = np.random.normal(mean_of_mean, std_of_mean, 1)
    scale = np.random.normal(mean_of_scale, std_of_scale, 1)
    for i in range(len(atmos)):
        std += np.random.normal(0, std_of_std, 1)
        while std < 150:
            std += np.random.normal(0, std_of_std, 1)
        mean += np.random.normal(0, std_of_mean, 1)
        while mean < 800:
            mean += np.random.normal(0, std_of_mean, 1)
        scale += np.random.normal(0, std_of_scale, 1)
        atmos[i] = 100 + scale * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(height-mean)**2/(2*std**2))
    f = interpolate.interp2d(range(0, 25, 1), height, np.transpose(atmos), kind='linear')
    atmos_full = f(time, height)
    return atmos_full
