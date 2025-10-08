import numpy as np
import scipy

class Uranus:

    #Physical parameters for Uranus in SI units:
    G                   = 6.67430e-11                   #https://arxiv.org/pdf/2409.03787, CODATA 2022
    M                   = 8.68099e+25                   #https://iopscience.iop.org/article/10.3847/1538-3881/ad99d1, Jacobson 2025 
    R                   = 25559*1e3                     #https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JA092iA13p14987, Lindal 1987
    R_type              = 'equatorial'                  #https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JA092iA13p14987, Lindal 1987
    P0                  = 1e5                           #https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JA092iA13p14987, Lindal 1987
    R_ref               = 25559*1e3                     #https://doi.org/10.1016/j.icarus.2024.115957, French 2024
    Prot                = 62064                         #https://www.nature.com/articles/322042a0, Desch 1986
    Target_Js           = [3509.291e-6, -35.522e-6]     #https://doi.org/10.1016/j.icarus.2024.115957, French 2024
    Sigma_Js            = [   0.412e-6,   0.466e-6]     #https://doi.org/10.1016/j.icarus.2024.115957, French 2024
    
    use_atmosphere      = True
    atmosphere_until    = 1e7
    
    alphas              = np.zeros(12)
    
    #alphas              = [1.77043*10**6, -1.08761*10**6, 231200, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #Target_Js           = [3510.68e-6, -34.17e-6]   #https://iopscience.iop.org/article/10.1088/0004-6256/148/5/76, Jacobsen
    #Sigma_Js            = [0.7e-06, 1.3e-06]        #https://iopscience.iop.org/article/10.1088/0004-6256/148/5/76, Jacobsen

    #https://arxiv.org/abs/1010.5546:
    H11         = np.loadtxt('planet_data/literature_values/Uranus/density_ravit_1.txt')
    H11_li      = H11[:, 0]                             #unitless
    H11_rhoi    = H11[:, 1]                             #1000 kg/m^3
    H11_Pi      = H11[:, 2]/1e9                         #GPa
    H11_Ti      = [np.nan]*len(H11[:, 0])               #K

    #https://arxiv.org/pdf/1207.2309
    N13_U1          = np.loadtxt('planet_data/literature_values/Uranus/table_U1.txt')
    N13_U2          = np.loadtxt('planet_data/literature_values/Uranus/table_U2.txt')
    N13_U1_li       = N13_U1[:, 2]/np.max(N13_U1[:, 2]) #unitless
    N13_U2_li       = N13_U2[:, 2]/np.max(N13_U1[:, 2]) #unitless
    N13_U1_rhoi     = N13_U1[:, 4]                      #1000 kg/m^3
    N13_U2_rhoi     = N13_U2[:, 4]                      #1000 kg/m^3
    N13_U1_Pi       = N13_U1[:, 1]                      #GPa
    N13_U2_Pi       = N13_U2[:, 1]                      #GPa
    N13_U1_Ti       = N13_U1[:, 3]                      #K
    N13_U2_Ti       = N13_U2[:, 3]                      #K

    #https://arxiv.org/abs/1908.10682:
    V20_U1      = np.loadtxt('planet_data/literature_values/Uranus/vazan_1.txt')
    V20_U2      = np.loadtxt('planet_data/literature_values/Uranus/vazan_2.txt')
    V20_U3      = np.loadtxt('planet_data/literature_values/Uranus/vazan_3.txt')
    V20_U4      = np.loadtxt('planet_data/literature_values/Uranus/vazan_4.txt')
    V20_U1_li   = V20_U1[:, 0]                          #unitless
    V20_U2_li   = V20_U2[:, 0]                          #unitless
    V20_U3_li   = V20_U3[:, 0]                          #unitless
    V20_U4_li   = V20_U4[:, 0]                          #unitless
    V20_U1_rhoi = V20_U1[:, 3]/1000                     #1000 kg/m^3
    V20_U2_rhoi = V20_U2[:, 3]/1000                     #1000 kg/m^3
    V20_U3_rhoi = V20_U3[:, 3]/1000                     #1000 kg/m^3
    V20_U4_rhoi = V20_U4[:, 3]/1000                     #1000 kg/m^3
    V20_U1_Pi   = V20_U1[:, 1]/1e9                      #GPa
    V20_U2_Pi   = V20_U2[:, 1]/1e9                      #GPa
    V20_U3_Pi   = V20_U3[:, 1]/1e9                      #GPa
    V20_U4_Pi   = V20_U4[:, 1]/1e9                      #GPa
    V20_U1_Ti   = V20_U1[:, 2]                          #K
    V20_U2_Ti   = V20_U2[:, 2]                          #K
    V20_U3_Ti   = V20_U3[:, 2]                          #K
    V20_U4_Ti   = V20_U4[:, 2]                          #K

    lis   = [H11_li,    N13_U1_li,     N13_U2_li,     V20_U1_li,      V20_U2_li,      V20_U3_li,      V20_U4_li]
    rhois = [H11_rhoi,  N13_U1_rhoi,   N13_U2_rhoi,   V20_U1_rhoi,    V20_U2_rhoi,    V20_U3_rhoi,    V20_U4_rhoi]
    Pis   = [H11_Pi,    N13_U1_Pi,     N13_U2_Pi,     V20_U1_Pi,      V20_U2_Pi,      V20_U3_Pi,      V20_U4_Pi]
    Tis   = [H11_Ti,    N13_U1_Ti,     N13_U2_Ti,     V20_U1_Ti,      V20_U2_Ti,      V20_U3_Ti,      V20_U4_Ti]
    label = ['H+10',    'N+13 U1',     'N+13 U2',     'VH20 U1',      'VH20 U2',      'VH20 U3',      'VH20 U4']

    #https://arxiv.org/abs/2111.15494:
    H20         = np.loadtxt('planet_data/literature_values/Uranus/Uranus_Clean_P_T_Rho_20solar_2_solar_NH3.dat', skiprows=1)
    H20_Pi      = H20[:, 0]*1e5       #Pa
    H20_Ti      = H20[:, 1]           #K
    H20_rhoi    = H20[:, 2]           #kg/m^3

    #Atmosphere:
    def polytrope(param, H20_rhoi=H20_rhoi, H20_Pi=H20_Pi):
        return np.sum(abs( H20_rhoi[H20_Pi<1e8] - (H20_Pi[H20_Pi<1e8]/param[0])**(param[1]/(param[1]+1)) ))

    #param = scipy.optimize.minimize(polytrope, np.array([2e5,1e0])).x
    param = [5.71682643e+05, 3.96776295e+00]

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