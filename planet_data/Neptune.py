import numpy as np
import scipy

class Neptune:

    #Physical parameters for Neptune in SI units:
    G                   = 6.67430e-11                   #https://arxiv.org/pdf/2409.03787, CODATA 2022
    M                   = 1.02409e+26                   #https://iopscience.iop.org/article/10.1088/0004-6256/137/5/4322/pdf, Jacobson 2009
    R                   = 24766*1e3                     #https://articles.adsabs.harvard.edu/pdf/1992AJ....103..967L, Lindal 1992
    R_type              = 'equatorial'                  #https://articles.adsabs.harvard.edu/pdf/1992AJ....103..967L, Lindal 1992
    P0                  = 1e5                           #https://articles.adsabs.harvard.edu/pdf/1992AJ....103..967L, Lindal 1992
    R_ref               = 25225*1e3                     #https://doi.org/10.1051/0004-6361/202244537, Wang 2023
    Prot                = 57479                         #https://doi.org/10.1016/j.icarus.2011.05.013, Karkoschka 2011
    Target_Js           = [3401.655e-6, -33.294e-6]     #https://doi.org/10.1051/0004-6361/202244537, Wang 2023
    Sigma_Js            = [   3.994e-6,  10.000e-6]     #https://doi.org/10.1051/0004-6361/202244537, Wang 2023

    use_atmosphere      = True
    atmosphere_until    = 1e7

    alphas              = np.zeros(12)

    #alphas              = [1346930, -5113950, 12592000, -23229300, 32883200, -35957200, 30052600, -18556900, 7933280, -2082600, 251643, 0] #my own work based of Smorovsky

    lis   = []
    rhois = []
    Pis   = []
    label = []

    #https://arxiv.org/abs/2111.15494:
    H20         = np.loadtxt('planet_data/literature_values/Neptune/Neptune_Clean_P_T_Rho_80solar_10solar.dat', skiprows=1)
    H20_Pi      = H20[:, 0]*1e5       #Pa
    H20_Ti      = H20[:, 1]           #K
    H20_rhoi    = H20[:, 2]           #kg/m^3 

    #https://arxiv.org/pdf/1207.2309
    N13_N1          = np.loadtxt('planet_data/literature_values/Neptune/table_N1.txt')
    N13_N2b         = np.loadtxt('planet_data/literature_values/Neptune/table_N2b.txt')
    N13_N1_li       = N13_N1[:, 2]*6380*1e3     #m
    N13_N2b_li      = N13_N2b[:, 2]*6380*1e3    #m
    N13_N1_rhoi     = N13_N1[:, 4]              #1000 kg/m^3
    N13_N2b_rhoi    = N13_N2b[:, 4]             #1000 kg/m^3
    N13_N1_Pi       = N13_N1[:, 1]              #GPa
    N13_N2b_Pi      = N13_N2b[:, 1]             #GPa
    N13_N1_Ti       = N13_N1[:, 3]              #K
    N13_N2b_Ti      = N13_N2b[:, 3]             #K

    lis   = [N13_N1_li,     N13_N2b_li]
    rhois = [N13_N1_rhoi,   N13_N2b_rhoi]
    Pis   = [N13_N1_Pi,     N13_N2b_Pi]
    Tis   = [N13_N1_Ti,     N13_N2b_Ti]
    label = ['N+13 N1',     'N+13 N2b']

    #Atmosphere:
    def polytrope(param, H20_rhoi=H20_rhoi, H20_Pi=H20_Pi):
        return np.sum(abs( H20_rhoi[H20_Pi<1e8] - (H20_Pi[H20_Pi<1e8]/param[0])**(param[1]/(param[1]+1)) ))

    #param = scipy.optimize.minimize(polytrope, np.array([2e5,1e0])).x
    param = [5.48841758e+05, 5.30367196e+00]

    def atmosphere(norm_radii, pressure, param=param):
        return (pressure/param[0])**(param[1]/(param[1]+1))

    """
    import matplotlib.pyplot as plt
    plt.figure(layout='constrained')
    plt.plot(H20_rhoi/1000, H20_Pi/1e5, label='Hueso et al.')
    plt.plot(atmosphere(None, H20_Pi, param=param)/1000, H20_Pi/1e5, label='polytrope fit')
    plt.xlabel(r'$\rho$ [g/cm$^3$]')
    plt.ylabel(r'$P$ [bar]')
    plt.xlim((0,0.1))
    plt.ylim((0,1000))
    plt.legend()
    plt.savefig('fit.png')
    """
    
