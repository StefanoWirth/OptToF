import numpy as np
import scipy

class Uranus:

    data = np.genfromtxt('planet_data/uranus4_luca.csv', delimiter=',', skip_header=1)

    R    = data[:,2] / 100
    P    = 10**data[:,3] * 0.1
    T    = 10**data[:,4]
    Rho  = 10**data[:,5] * 1000

    #Physical parameters for Uranus in SI units:
    G                   = 6.6743e-11 
    M                   = 86.8e24    #http://dx.doi.org/10.1098/rsta.2019.0474
    R_eq                = R[0]*1.0058238550027847
    Prot                = 62040         #Saburo
    P0                  = P[0]
    #Target_Js           = [3510.68e-6, -34.17e-6]   #https://iopscience.iop.org/article/10.1088/0004-6256/148/5/76, Jacobsen
    #Sigma_Js            = [0.7e-06, 1.3e-06]        #https://iopscience.iop.org/article/10.1088/0004-6256/148/5/76, Jacobsen
    Target_Js           = [3509.291e-6, -35.522e-6]  #https://doi.org/10.1016/j.icarus.2024.115957, French
    Sigma_Js            = [0.412e-6, 0.466e-6]       #https://doi.org/10.1016/j.icarus.2024.115957, French
    use_atmosphere      = True
    atmosphere_until    = P[-20]
    alphas              = [1.77043*10**6, -1.08761*10**6, 231200, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    #My work:
    LM          = np.loadtxt('planet_data/minmax.txt')
    LM_li       = LM[0, :] #m
    LM_rhoi_min = LM[1, :] #1000 kg/m^3
    LM_rhoi_max = LM[2, :] #1000 kg/m^3

    lis     = [LM_li,                 LM_li]
    rhois   = [LM_rhoi_min,           LM_rhoi_max]
    Pis     = [np.zeros_like(LM_li),  np.zeros_like(LM_li)]
    Tis     = [np.zeros_like(LM_li),  np.zeros_like(LM_li)]
    labels  = ['Uranus min.',         'Uranus max.']
    colors  = ['C0', 'C1']
    lws     = [2, 2]
    lss     = ['-', '-']


    func = scipy.interpolate.interp1d(R/np.max(R), Rho, kind='linear', fill_value="extrapolate")
    
    #Atmosphere:
    def atmosphere(radii, pressure, func=func):
        
        return func(radii/np.max(radii))
