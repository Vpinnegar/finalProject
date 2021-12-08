import numpy as np
import miepython
import sys

def MieBackscatter(time,height, inpu, m, lambd, typ):
    back = np.zeros((height, time))
    K = 1.38e-23
    if typ == 'cloud':
        for sec in range(time):
            sys.stdout.write('\rCalculating time bin {:<4d} of {:<4d}'.format(sec+1, time))
            sys.stdout.flush()
            for al in range(height):
                if inpu[al,sec] >= 300:
                    particler = np.random.normal(5e-7, 1e-7, 1) * inpu[al,sec]
                    x = 2*np.pi*particler/lambd
                    _, _, Qback, _ = miepython.mie(m,x)
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                elif 299 > inpu[al,sec] >= 250: 
                    particler = np.random.normal(5e-8, 1e-8, 1) * inpu[al,sec]
                    x = 2*np.pi*particler/lambd
                    _, _, Qback, _ = miepython.mie(m,x)
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                elif 250 > inpu[al,sec] >= 200: 
                    particler = np.random.normal(5e-9, 1e-9, 1) * inpu[al,sec]
                    x = 2*np.pi*particler/lambd
                    _, _, Qback, _ = miepython.mie(m,x)
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                else:
                    if lambd == 532e-9:
                        n = 1.000278
                    elif lambd == 8.6e-3:
                        n = 1.000273
                    particler = np.random.normal(5e-11, 1e-11, 1) * inpu[al,sec]
                    bigboy = np.abs(((n**2 + 2)/(n**2 - 1)))**2
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* particler**6 * lambd**(-4)
    elif typ == 'aerosol':
        for sec in range(time):
            sys.stdout.write('\rCalculating time bin {:<4d} of {:<4d}'.format(sec+1, time))
            sys.stdout.flush()
            for al in range(height):
                if inpu[al,sec] >= 300:
                    particler = np.random.normal(5e-11, 1e-11, 1) * inpu[al,sec]
                    x = 2*np.pi*particler/lambd
                    _, _, Qback, _ = miepython.mie(m,x)
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                elif 299 > inpu[al,sec] >= 250: 
                    particler = np.random.normal(5e-12, 1e-12, 1) * inpu[al,sec]
                    x = 2*np.pi*particler/lambd
                    _, _, Qback, _ = miepython.mie(m,x)
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                elif 250 > inpu[al,sec] >= 200: 
                    particler = np.random.normal(5e-13, 1e-13, 1) * inpu[al,sec]
                    x = 2*np.pi*particler/lambd
                    _, _, Qback, _ = miepython.mie(m,x)
                    back[al,sec] = (inpu[al,sec]/K)*np.pi*(particler**2)*Qback
                else:
                    if lambd == 532e-9:
                        n = 1.000278
                    elif lambd == 8.6e-3:
                        n = 1.000273
                    particler = np.random.normal(5e-15, 1e-15, 1) * inpu[al,sec]
                    bigboy = np.abs(((n**2 + 2)/(n**2 - 1)))**2
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* particler**6 * lambd**(-4)       
    print()
    return back

def RadBackscatter(time,height, inpu, n, wave, typ):
    back = np.zeros((height, time))
    K = 1.38e-23
    bigboy = np.abs(((n**2 + 2)/(n**2 - 1)))**2
    if typ == 'cloud':
        for sec in range(time):
            sys.stdout.write('\rCalculating time bin {:<4d} of {:<4d}'.format(sec+1, time))
            sys.stdout.flush()
            for al in range(height):
                if inpu[al,sec] >= 300:
                    r = np.random.normal(5e-7, 1e-7, 1) * inpu[al,sec]
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                elif 299 > inpu[al,sec] >= 250: 
                    r = np.random.normal(5e-8, 1e-8, 1) * inpu[al,sec]
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                elif 250 > inpu[al,sec] >= 200: 
                    r = np.random.normal(5e-9, 1e-9, 1) * inpu[al,sec]
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                else: 
                    r = np.random.normal(5e-11, 1e-11, 1) * inpu[al,sec]
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)	
    elif typ == 'aerosol':
        for sec in range(time):
            sys.stdout.write('\rCalculating time bin {:<4d} of {:<4d}'.format(sec+1, time))
            sys.stdout.flush()
            for al in range(height):
                if inpu[al,sec] >= 300:
                    r = np.random.normal(5e-11, 1e-11, 1) * inpu[al,sec]
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                elif 299 > inpu[al,sec] >= 250: 
                    r = np.random.normal(5e-12, 1e-12, 1) * inpu[al,sec]
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                elif 250 > inpu[al,sec] >= 200: 
                    r = np.random.normal(5e-13, 1e-13, 1) * inpu[al,sec]
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
                else: 
                    r = np.random.normal(5e-15, 1e-15, 1) * inpu[al,sec]
                    back[al,sec] = (inpu[al,sec]/K) *bigboy* r**6 * wave**(-4)
    print()
    return back

# def VEGABetaMol(time, height, inpu, wave):
#     betamol = np.zeros((height, time))
#     K = 1.38e-23
#     for i in range(time):
#         for j in range(height):
#             betamol[j,i] = (inpu[j,i] /K) *5.31e-22* wave**(-4)
#     return betamol

# def ReflectivitytoBackscatter(Z, m, wavelength):
#     for i in range(len(Z)):
#         beta = ((m**2 -1)/(m**2 + 2))*(np.pi**4/(4*wavelength**4))*Z[i]
#     return beta

# def BackscattertoReflectivity(B, m, wavelength):
#     for i in range(len(Z)):
#         beta = ((m**2 -1)/(m**2 + 2))*(np.pi**4/(4*wavelength**4))*Z[i]
#     return beta

# def Lidar(Power, binlength, Area, Eff, Range, Beta, Transmission):
#     P = []
#     numbins = Range/binlength
#     perkm = numbins/ (Range/1000)
#     overlap = np.ones(Range)
#     km1 = np.linspace(0,1,int(perkm))
#     for i in range(len(km1)):
#         overlap[i] = km1[i]
#     for height in range(len(Beta)):
#         PH = Power * (binlength) * Area * Eff * (1/(Range**2))*Beta[height]*Transmission
#         P.append(PH)
#     return P

# def Radar(Power, binlength, Gain, theta, phi, K, loss, Beta,\
#  wavelength, Range, m):
#     Z_e = Beta * (m**2-1)/(m**2+2) * 4 * wavelength**4 / np.pi**4
#     P = np.pi**3 * Power * Gain**2 * theta*phi * binlength * K**2 * loss * Z_e
#     return P
