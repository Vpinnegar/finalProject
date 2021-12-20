import numpy as np
import miepython
import sys

def MieBackscatter(time, height, inpu, m, lambd, typ, particle_radii):
    back = np.zeros((height, time))
    K = 1.38e-23
    x = 2*np.pi*particle_radii/lambd
    for sec in range(time):
        sys.stdout.write('\rCalculating time bin {:<4d} of {:<4d}'.format(sec+1, time))
        sys.stdout.flush()
        for al in range(height):
            if inpu[al,sec] >= 200:
                _, _, Qback, _ = miepython.mie(m,x[al,sec])
                back[al,sec] = (inpu[al,sec]/K)*np.pi*(particle_radii[al,sec]**2)*Qback
            elif 200 > inpu[al,sec]:
                if lambd == 532e-9:
                    n = 1.000278
                elif lambd == 8.6e-3:
                    n = 1.000273
                bigboy = np.abs(((n**2 + 2)/(n**2 - 1)))**2
                back[al,sec] = (inpu[al,sec]/K) *bigboy* particle_radii[al,sec]**6 * lambd**(-4)      
    print()
    return back

def RadBackscatter(time, height, inpu, n, wave, typ, particle_radii):
    back = np.zeros((height, time))
    K = 1.38e-23
    bigboy = np.abs(((n**2 + 2)/(n**2 - 1)))**2
    for sec in range(time):
        sys.stdout.write('\rCalculating time bin {:<4d} of {:<4d}'.format(sec+1, time))
        sys.stdout.flush()
        for al in range(height):
            if inpu[al,sec] >= 200:
                back[al,sec] = (inpu[al,sec]/K) *bigboy* particle_radii[al,sec]**6 * wave**(-4)
            else:
                if wave == 532e-9:
                    m = 1.000278
                elif wave == 8.6e-3:
                    m = 1.000273
                bigboy_2 = np.abs(((m**2 + 2)/(m**2 - 1)))**2 
                back[al,sec] = (inpu[al,sec]/K) *bigboy_2* particle_radii[al,sec]**6 * wave**(-4)	
    print()
    return back