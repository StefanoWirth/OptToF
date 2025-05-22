from ClassToF import ToF
from ClassOpt import OptToF

"""
This process runs at a cost of 45 KB per minute per core (Intel Core i7-8650U @ 2.40GHz)
Estimated cost for 24 hours at 64 cores: ~8.3GB, assuming cores to be twice as good as mine 
"""

if __name__ == '__main__':

    kwargs = {}

    kwargs['N']      = 2**10
    kwargs['G']      = 6.6743e-11
    kwargs['M_init'] = 86.8127e24
    kwargs['R_init'] = 25559*1e3
    kwargs['Period'] = 62080
    kwargs['P0']     = 1e5
    kwargs['Target_Js'] = [3509.291e-6, -35.522e-6] #https://doi.org/10.1051/0004-6361/202244537
    kwargs['Sigma_Js']  = [0.412e-6, 0.466e-6] #https://doi.org/10.1051/0004-6361/202244537
    kwargs['verbosity'] = 0
    kwargs['MaxIterHE'] = 100
    kwargs['use_atmosphere'] = False
    kwargs['atmosphere'] = None
    kwargs['atmosphere_until'] = None

    X = ToF(**kwargs)

    kwargs = {}

    kwargs['verbosity'] = 0
    kwargs['time'] = False
    kwargs['parallelize'] = True
    kwargs['steps'] = 3000
    kwargs['learning rate'] = 0.05  #This learning rate has been revealed to me by god
    kwargs['costfactor'] = 1e5
    kwargs['ToF convergence tolerance'] = 1e-6
    kwargs['figures'] = False
    kwargs['kitty'] = 0
    kwargs['write to file'] = True
    kwargs['continuous running'] = True
    kwargs['cores'] = 6
    kwargs['file location'] = 'bigrun_uranus_uniform.hdf5'
    kwargs['minimum increase'] = 1

    Optimiser = OptToF(**kwargs)

    Optimiser.run(X)