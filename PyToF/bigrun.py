from ClassToF import ToF
from ClassOpt import OptToF
from planet_data.Uranus import Uranus
from planet_data.Neptune import Neptune

"""
This process runs at a cost of 45 KB per minute per core (Intel Core i7-8650U @ 2.40GHz)
Estimated cost for 24 hours at 64 cores: ~8.3GB, assuming cores to be twice as good as mine 
"""

if __name__ == '__main__':

    kwargs = {}

    kwargs['N']      = 2**10
    kwargs['verbosity'] = 0
    kwargs['MaxIterShape'] = 100
    kwargs['use_atmosphere'] = False
    kwargs['atmosphere'] = None
    kwargs['atmosphere_until'] = None

    #Uranus

    kwargs['G']      = Uranus.G
    kwargs['M_phys'] = Uranus.M
    kwargs['R_ref'] = Uranus.R_ref
    kwargs['R_phys'] = [Uranus.R, Uranus.R_type]
    kwargs['Period'] = Uranus.Prot
    kwargs['P0']     = 0
    kwargs['Target_Js'] = Uranus.Target_Js
    kwargs['Sigma_Js']  = Uranus.Sigma_Js

    UranusToF = ToF(**kwargs)

    #Neptune

    kwargs['G']      = Neptune.G
    kwargs['M_phys'] = Neptune.M
    kwargs['R_ref'] = Neptune.R_ref
    kwargs['R_phys'] = [Neptune.R, Neptune.R_type]
    kwargs['Period'] = Neptune.Prot
    kwargs['P0']     = 0
    kwargs['Target_Js'] = Neptune.Target_Js
    kwargs['Sigma_Js']  = Neptune.Sigma_Js

    NeptuneToF = ToF(**kwargs)

    kwargs = {}

    kwargs['verbosity'] = 3
    kwargs['time'] = True
    kwargs['parallelize'] = True
    kwargs['steps'] = 3000
    kwargs['epoch size'] = 25
    kwargs['learning rate'] = 0.08  #This learning rate has been revealed to me by god
    kwargs['costfactor'] = 1e2
    kwargs['ToF convergence tolerance'] = 1e-6
    kwargs['figures'] = True
    kwargs['kitty'] = 0
    kwargs['write to file'] = True
    kwargs['continuous running'] = False
    kwargs['cores'] = 8
    kwargs['file location'] = 'newrun_uranus.hdf5' #TODO: MODIFY
    kwargs['minimum increase'] = 1
    kwargs['DBGshowchance'] = 0
    #print('learning rate: '+str(kwargs['learning rate']))
    #print('cost factor: '+str(kwargs['costfactor']))
    Optimiser = OptToF(**kwargs)

    Optimiser.run(UranusToF) #TODO: MODIFY