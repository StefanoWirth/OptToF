from ClassToF import ToF
from ClassOpt import OptToF

"""
This process runs at a cost of 45 KB per minute per core (Intel Core i7-8650U @ 2.40GHz)
Estimated cost for 24 hours at 64 cores: ~8.3GB, assuming cores to be twice as good as mine 
"""

if __name__ == '__main__':

    kwargs = {}

    #Neptune #TODO: Change to Uranus values for Uranus running, also are these values good?
    kwargs['N']      = 2**10
    kwargs['G']      = 6.6743e-11
    kwargs['M_init'] = 6836525.21*(1000)**3/kwargs['G'] #https://doi.org/10.1051/0004-6361/202244537
    kwargs['R_init'] = 25225*1e3                        #https://iopscience.iop.org/article/10.1088/0004-6256/137/5/4322
    kwargs['Period'] = 15.9663*60*60                    #https://www.sciencedirect.com/science/article/pii/S0019103511001783?via%3Dihub
    kwargs['P0']     = 1e5
    kwargs['Target_Js'] = [3401.655e-6, -33.294e-6] #https://doi.org/10.1051/0004-6361/202244537
    kwargs['Sigma_Js']  = [   3.994e-6,  10.000e-6] #https://doi.org/10.1051/0004-6361/202244537
    kwargs['verbosity'] = 0
    kwargs['MaxIterHE'] = 100
    kwargs['use_atmosphere'] = False #TODO: these are supported but untested, can use if you want to
    kwargs['atmosphere'] = None
    kwargs['atmosphere_until'] = None

    X = ToF(**kwargs)

    kwargs = {}

    kwargs['verbosity'] = 0
    kwargs['time'] = False
    kwargs['parallelize'] = True
    kwargs['steps'] = 3000
    kwargs['learning rate'] = 0.05
    kwargs['costfactor'] = 1e5
    kwargs['ToF convergence tolerance'] = 1e-6
    kwargs['figures'] = False
    kwargs['kitty'] = 0
    kwargs['write to file'] = True
    kwargs['continuous running'] = True
    kwargs['cores'] = 8 #TODO
    kwargs['file location'] = 'bigrun_neptune.hdf5' #TODO: bigrun_uranus.hdf5
    kwargs['minimum increase'] = 1 #TODO #minimum increase of rho at each step in SI units, can be a scalar or an np.array of shape params (N-1) (because N-1 increases)

    Optimiser = OptToF(**kwargs)

    Optimiser.run(X)