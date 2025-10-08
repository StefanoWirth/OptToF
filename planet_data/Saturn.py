import numpy as np
import scipy

class Saturn:

    #Physical parameters for Saturn in SI units:
    #https://arxiv.org/abs/2202.10046:
    G                   = 6.6743e-11
    M                   = 568.319e24
    R_eq                = 60268*1e3
    Prot                = 10*60*60 + 39*60
    P0                  = 1e5
    #10.1126/science.aat2965 (2019):
    Target_Js           = [16290.573e-6, -935.314e-6, 86.340e-6]#,  -14.624e-6]
    #https://doi.org/10.1002/2017GL073629:
    Sigma_Js            = [120*0.27e-6,  14*0.36e-6,  11*0.24e-6]#, 3*0.31e-6]
    #print('required accuracy', np.array(Sigma_Js)/np.array(Target_Js))
    use_atmosphere      = True
    atmosphere_until    = 1e7

    lis   = []
    rhois = []
    Pis   = []
    label = []

    #Atmosphere:
    #https://doi.org/10.3847/1538-4357/ab71ff:
    Q_params = [30080.0859955084, -112840.747945798, 158672.644582175, -99142.6945670244, 23230.7164574213]

    def atmosphere(normalized_radii, Q_params=Q_params):
        res = 0
        for i in range(len(Q_params)):
            res += Q_params[i]*normalized_radii**(4-i)*0.12828*1000
        if len(res)>2:
            res[1] += np.diff(res)[1] #no density inversions
        if len(res)>1:
            res[0] += np.diff(res)[0] #no density inversions
        return res
