import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.transforms as trans
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotview import inset_zoom_axes

"""
IMPORTANT NOTICE:
    This code is horrible, completely undocumented, arcane, obscure, terribly implemented, not reusable,
    and essentially the textbook definition of technical debt.
    I do not care. Good luck 👍.
"""

plt.rcParams.update({'font.size': 17})
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.max_open_warning'] = False

N_uranus = 1277081
N_neptune = 1341762

N = 2**10    #size of datasets
RES = 4*N

is_neptune = False

def generate_plots():
    tic = time.perf_counter()
    global jumpcriterion, jumpcriteria
    jumpcriterion = 100
    #jumpcriteria = [50,75,100,150,250,500,1000]
    jumpcriteria = np.arange(50, 1001, 50)

    if is_neptune:
        with open('result_dict_neptune.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        with open('result_dict_uranus.pkl', 'rb') as f:
            results = pickle.load(f)


    toc = time.perf_counter()



    #PLOT ================================

    fig, axs = plt.subplots(5, 5, layout = 'constrained', num = str(is_neptune))

    if is_neptune == True:
        planetname = 'neptune'
        plottitlestr = ' (Neptune)'
        defaultcolor = '#3b55e1'
    else:
        planetname = 'uranus'
        plottitlestr = ' (Uranus)'
        defaultcolor = '#50b5ad'

    savefig = [[None for i in range(5)] for j in range(5)]
    saveaxs = [[None for i in range(5)] for j in range(5)]

    for i in range(5):
        for j in range(5):
            savefig[i][j], saveaxs[i][j] = plt.subplots(num = str([i,j]))

    dpi = 200

    bincount = N    #THE BINS MARTY THE BINS WHAT DO THEY MEAN

    nticks = 5

    xticklocations = np.linspace(1023, 0, nticks)
    xticklabels = np.linspace(0, 1, nticks)
    yticklocations = [0, 5000, 10000, 15000, 20000]
    yticklabels = [0, 5, 10, 15, 20]

    #JUMPS

    #saveaxs[0][0].set_title('Average jump' + plottitlestr)
    saveaxs[0][0].bar(range(1023), results['raw jumps expl'], width=1, color = defaultcolor)
    saveaxs[0][0].invert_xaxis()
    saveaxs[0][0].set_xticks(xticklocations, xticklabels)
    saveaxs[0][0].set_xlabel("Normalised radius $r/R$")
    saveaxs[0][0].set_ylabel(r"$\overline{\Delta\rho}$ [kg m$^{-3}$]")

    savefig[0][0].savefig('plots/' + planetname + '/00rawjumps' + '_' + planetname + '.png', dpi=dpi)

    """
   # saveaxs[0,1].set_title('Raw Jumps of J explained (averaged)')
    saveaxs[0,1].plot(np.convolve(results['raw jumps expl'], np.ones(5)/5, mode = 'same'))
    """
    #saveaxs[0][1].set_title('Average jump intensity' + plottitlestr) #((real_jumplocation, jumpsize, jumpmagnitude))
    proc_jump_j_expl = results['processed jumps'][results['jumps Js explained view']]
    jumpindex = proc_jump_j_expl[:, 0]
    jumpwidth = proc_jump_j_expl[:, 1]
    jumpheight = proc_jump_j_expl[:, 2]/results['nJ']
    jumpweight = jumpheight/jumpwidth
    saveaxs[0][1].hist(jumpindex, bins = bincount, range = (0,N-1), weights = jumpweight, color = defaultcolor)
    saveaxs[0][1].invert_xaxis()
    saveaxs[0][1].set_xticks(xticklocations, xticklabels)
    saveaxs[0][1].set_xlabel("Normalised radius $r/R$")
    saveaxs[0][1].set_ylabel(r"$\overline{\frac{\Delta\rho}{\Delta l}}$  [1/$\Delta l$ kg m$^{-3}$]")

    savefig[0][1].savefig('plots/' + planetname + '/01procjumps1' + '_' + planetname + '.png', dpi=dpi)

    #saveaxs[0][2].set_title('Jump widths, arbitrary units' + plottitlestr) #((real_jumplocation, jumpsize, jumpmagnitude))
    saveaxs[0][2].hist(jumpindex, bins = bincount, range = (0,N-1), weights = jumpwidth, color = defaultcolor)
    saveaxs[0][2].invert_xaxis()
    saveaxs[0][2].set_xticks(xticklocations, xticklabels)
    saveaxs[0][2].set_xlabel("Normalised radius $r/R$")
    saveaxs[0][2].get_yaxis().set_ticks([])

    savefig[0][2].savefig('plots/' + planetname + '/02procjumps2' + '_' + planetname + '.png', dpi=dpi)

    #saveaxs[0][3].set_title('Average jump magnitude' + plottitlestr) #((real_jumplocation, jumpsize, jumpmagnitude))
    saveaxs[0][3].hist(jumpindex, bins = bincount, range = (0,N-1), weights = jumpheight, color = defaultcolor)
    saveaxs[0][3].invert_xaxis()
    saveaxs[0][3].set_xticks(xticklocations, xticklabels)
    saveaxs[0][3].set_xlabel("Normalised radius $r/R$")
    saveaxs[0][3].set_ylabel(r"$\overline{\Delta\rho}$ [kg m$^{-3}$]")

    savefig[0][3].savefig('plots/' + planetname + '/03procjumps3' + '_' + planetname + '.png', dpi=dpi)

    #saveaxs[1][2].set_title('Number of jumps in profile' + plottitlestr)
    nr_jumps_J_expl = results['nr jumps'][results['nr jumps Js explained view']]
    max_nr = np.max(nr_jumps_J_expl)
    nrjumpsbinvalues, nrjumpsbinedges, _ = saveaxs[1][2].hist(nr_jumps_J_expl, bins = max_nr + 1, range = (-0.5, max_nr+0.5), color = defaultcolor, density = True)
    #find index where we cumulate 95% of the total nr of jumps
    transform = trans.blended_transform_factory(saveaxs[1][2].transData, saveaxs[1][2].transAxes)
    saveaxs[1][2].vlines(np.percentile(nr_jumps_J_expl, 50), ymin = 0, ymax = 1, linestyle ='-', color = 'black', transform = transform, label = 'Median')
    saveaxs[1][2].vlines(np.percentile(nr_jumps_J_expl, 95), ymin = 0, ymax = 1, linestyle ='--', color = 'tab:gray', transform = transform, label = '95th percentile')
    saveaxs[1][2].vlines(np.mean(nr_jumps_J_expl), ymin = 0, ymax = 1, linestyle ='-', color = 'tab:red', transform = transform, label = 'Average')
    saveaxs[1][2].xaxis.set_major_locator(MaxNLocator(integer=True))
    saveaxs[1][2].set_xlabel("Number of jumps")
    saveaxs[1][2].set_ylabel("Relative frequency")
    saveaxs[1][2].get_yaxis().set_ticks([])
    saveaxs[1][2].legend()
    #savefig[1][2].subplots_adjust(left = 0.15)

    savefig[1][2].savefig('plots/' + planetname + '/12njumpexpl' + '_' + planetname + '.png', dpi=dpi)
    
    #saveaxs[1][3].set_title('Number of jumps in profile per criterion'+plottitlestr+'\nAverage and 95th percentile')
    jumps_per_criteria = results['nr jumps per criterion']
    jumpcriteria # x           
    avg_jumps_per_criteria = [] # y1
    max_jumps_per_criteria = [] # y2
    for jumps in jumps_per_criteria:
        avg_jumps_per_criteria.append(np.mean(jumps))
        max_jumps_per_criteria.append(np.percentile(jumps, 95))        #find index which covers 95th percentile
    saveaxs[1][3].plot(avg_jumps_per_criteria, color = 'tab:red', label = 'Average')
    saveaxs[1][3].plot(max_jumps_per_criteria, linestyle ='--', color = 'tab:gray', label = '95th percentile')
    saveaxs[1][3].legend()
    xticks = range(len(jumpcriteria))[::2]
    saveaxs[1][3].set_xticks(xticks)
    saveaxs[1][3].set_xticklabels(jumpcriteria[xticks])
    xticks = [0,1,3,5,7,9,11,13,15,17,19]
    xticks = [0,4,9,14,19]
    saveaxs[1][3].set_xticks(xticks)
    saveaxs[1][3].set_xticklabels(jumpcriteria[xticks])
    saveaxs[1][3].set_xlabel(r"Jumpcriteria $\rho/\Delta l$ [1/$\Delta l$ kg m$^{-3}$]")
    saveaxs[1][3].set_ylabel("Number of jumps")

    #for i in range(len(jumpcriteria)):
        #print(str(jumpcriteria[i]) + '&' + str(round(avg_jumps_per_criteria[i])) + '&' + str(int(max_jumps_per_criteria[i])) + '\\\\')
    #print(jumpcriteria)
    #print(avg_jumps_per_criteria)
    #print(max_jumps_per_criteria)
    savefig[1][3].savefig('plots/' + planetname + '/13jumpspercriteria' + '_' + planetname + '.png', dpi=dpi)

    """
    plt.title('All the ones that failed to explain J')
    for rho in results['J failures']:
        plt.plot(rho, alpha = 0.1)
    """

    #DISTR & CONTOUR

    #saveaxs[1][0].set_title('Distribution of density profiles, log scale' + plottitlestr)
    log_grid = results['distr grid density']
    logdistr = saveaxs[1][0].imshow((np.transpose(log_grid)), cmap = 'plasma', norm = col.LogNorm(), aspect = 'auto', origin = 'lower', extent = (-0.5, N-0.5, -0.5, 2e4-0.5))
    cbar = savefig[1][0].colorbar(logdistr, ax=saveaxs[1][0])
    cbar.set_label(label = 'Relative frequency', labelpad = 14)

    #literature values
    x = np.linspace(1,0,1024)
    if is_neptune:
        nettelmanNone = 1000*np.flip(np.loadtxt(r'planet_data\literature_values\Neptune\table_N1.txt')[:,4])
        nettelmanNonex = np.flip(np.loadtxt(r'planet_data\literature_values\Neptune\table_N1.txt')[:,2])
        nettelmanNonex /= nettelmanNonex[-1]
        nettelmanNtwo = 1000*np.flip(np.loadtxt(r'planet_data\literature_values\Neptune\table_N2b.txt')[:,4])
        nettelmanNtwox = np.flip(np.loadtxt(r'planet_data\literature_values\Neptune\table_N2b.txt')[:,2])
        nettelmanNtwox /= nettelmanNtwox[-1]
        saveaxs[1][0].plot(np.interp(x, nettelmanNonex, nettelmanNone), linestyle ='-', linewidth = 1, color = 'xkcd:marigold')
        saveaxs[1][0].plot(np.interp(x, nettelmanNtwox, nettelmanNtwo), linestyle ='--', linewidth = 1, color = 'xkcd:marigold')

    else: 
        nettelmanUone = 1000*np.flip(np.loadtxt(r'planet_data\literature_values\Uranus\table_U1.txt')[:,4])
        nettelmanUonex = np.flip(np.loadtxt(r'planet_data\literature_values\Uranus\table_U1.txt')[:,2])
        nettelmanUonex /= nettelmanUonex[-1]
        nettelmanUtwo = 1000*np.flip(np.loadtxt(r'planet_data\literature_values\Uranus\table_U2.txt')[:,4])
        nettelmanUtwox = np.flip(np.loadtxt(r'planet_data\literature_values\Uranus\table_U2.txt')[:,2])
        nettelmanUtwox /= nettelmanUtwox[-1]
        saveaxs[1][0].plot(np.interp(x, nettelmanUonex, nettelmanUone), linestyle ='-', linewidth = 1, color = 'xkcd:marigold')
        saveaxs[1][0].plot(np.interp(x, nettelmanUtwox, nettelmanUtwo), linestyle ='--', linewidth = 1, color = 'xkcd:marigold')
        helled = 1000*np.flip(np.loadtxt(r'planet_data\literature_values\Uranus\density_ravit_1.txt')[:,1])
        helledx = np.flip(np.loadtxt(r'planet_data\literature_values\Uranus\density_ravit_1.txt')[:,0])
        saveaxs[1][0].plot(np.interp(x, helledx, helled), linestyle ='-', linewidth = 1, color = 'xkcd:deep red')
        vazantwo = (np.loadtxt(r'planet_data\literature_values\Uranus\vazan_2.txt')[:,3])
        vazantwox = (np.loadtxt(r'planet_data\literature_values\Uranus\vazan_2.txt')[:,0])
        vazanthree = (np.loadtxt(r'planet_data\literature_values\Uranus\vazan_3.txt')[:,3])
        vazanthreex = (np.loadtxt(r'planet_data\literature_values\Uranus\vazan_3.txt')[:,0])
        saveaxs[1][0].plot(np.interp(x, vazantwox, vazantwo), linestyle ='-', linewidth = 1, color = 'xkcd:sky blue')
        saveaxs[1][0].plot(np.interp(x, vazanthreex, vazanthree), linestyle ='--', linewidth = 1, color = 'xkcd:sky blue')


    saveaxs[1][0].invert_xaxis()
    saveaxs[1][0].set_xticks(xticklocations, xticklabels)
    saveaxs[1][0].set_xlabel("Normalised radius $r/R$")
    saveaxs[1][0].set_ylim(top = 20000)
    saveaxs[1][0].locator_params(axis='y', nbins=5)
    saveaxs[1][0].set_yticks(yticklocations, yticklabels)
    saveaxs[1][0].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")
    #axins = zoomed_inset_axes(saveaxs[1][0], zoom = 2, loc='upper right')
    axins = inset_zoom_axes(saveaxs[1][0], [0.5, 0.5, 0.45, 0.45])
    #axins.imshow((np.transpose(log_grid)), cmap = 'plasma', norm = col.LogNorm(), aspect = 'auto', origin = 'lower', extent = (-0.5, N-0.5, -0.5, 2e4-0.5))
    axins.set_xticks([0,100,200,300], [1.0,0.9,0.8,0.7])
    axins.set_xlim(300,0)
    axins.set_ylim(0,2000)
    axins.set_yticks([0, 1000, 2000], [0, 1, 2])



    savefig[1][0].savefig('plots/' + planetname + '/10dens_distr' + '_' + planetname + '.png', dpi=2*dpi)


    #saveaxs[1][1].set_title('Distribution of density profiles, contour' + plottitlestr)
    percentile_grid = np.zeros_like(results['distr grid density'])
    contourintervals = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    cumulative_distr = np.cumsum(results['distr grid density'], axis=1)
    twolowersd      = -np.ones(N) #at 2.5%
    lowersd         = -np.ones(N) #at 16%
    contourmedian   = -np.ones(N) #at 50%
    uppersd         = -np.ones(N) #at 84%
    twouppersd      = -np.ones(N) #at 97.5%

    for x in range(N):
        level = 0
        y = 0
        while level < 10:
            while cumulative_distr[x, y] < 0.5-contourintervals[level]/2:
                percentile_grid[x, y] += level
                if cumulative_distr[x, y] > 0.025 and twolowersd[x] == -1: twolowersd[x] = y*2e4/RES
                if cumulative_distr[x, y] > 0.16  and lowersd[x] == -1: lowersd[x] = y*2e4/RES
                y += 1
            level += 1
        level -= 1
        while level > -1:
            while cumulative_distr[x, y] < 0.5+contourintervals[level]/2:
                percentile_grid[x, y] += level + 1
                if cumulative_distr[x, y] > 0.5   and contourmedian[x] == -1: contourmedian[x] = y*2e4/RES
                if cumulative_distr[x, y] > 0.84  and uppersd[x] == -1: uppersd[x] = y*2e4/RES
                y += 1
            level -= 1
        twouppersd[x] = y*2e4/RES
        """
        for y in range(RES):
            cumulative += results['distr grid density'][x, y]
            for interval in contourintervals:
                if cumulative >= 0.5-interval/2 and cumulative <= 0.5+interval/2:
                    percentile_grid[x, y] += 1
        """
    boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    percentile_grid = np.transpose(percentile_grid)
    cont_colours = plt.get_cmap(name='plasma')(np.linspace(0,1,11))
    cont_colours[0] = [1, 1, 1, 1] #white
    cont = saveaxs[1][1].contourf(results['x'], results['y'], percentile_grid, levels = boundaries, colors=cont_colours) 
    cbar = savefig[1][1].colorbar(cont, ax=saveaxs[1][1], ticks = boundaries)
    cbar.ax.set_yticklabels(['100%', '95%', '90%','80%','70%','60%','50%','40%','30%','20%','10%','0%'])
    cbar.set_label(label = 'Interval')
    #saveaxs[1][1].plot(twolowersd    , color = 'black', linewidth = 0.5, linestyle = (0, (2, 2)), label = '2nd percentile')
    saveaxs[1][1].plot(lowersd       , color = 'black', linewidth = 1, linestyle = '--')#, label = '16th percentile')
    saveaxs[1][1].plot(contourmedian , color = 'black', linewidth = 1, label = 'Median')
    saveaxs[1][1].plot(uppersd       , color = 'black', linewidth = 1, linestyle = '--', label = r'$\pm\sigma$')
    #saveaxs[1][1].plot(twouppersd    , color = 'black', linewidth = 0.5, linestyle = (0, (2, 2)), label = '98th percentile')
    saveaxs[1][1].invert_xaxis()
    saveaxs[1][1].set_xticks(xticklocations, xticklabels)
    saveaxs[1][1].set_xlabel("Normalised radius $r/R$")
    saveaxs[1][1].locator_params(axis='y', nbins=5)
    saveaxs[1][1].set_yticks(yticklocations, yticklabels)
    saveaxs[1][1].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")
    axins = inset_zoom_axes(saveaxs[1][1], [0.5, 0.5, 0.45, 0.45])
    axins.set_xticks([0,100,200,300], [1.0,0.9,0.8,0.7])
    axins.set_xlim(300,0)
    axins.set_ylim(0,2000)
    axins.set_yticks([0, 1000, 2000], [0, 1, 2])
    saveaxs[1][1].legend(loc = 'upper left', fontsize = 14)

    savefig[1][1].savefig('plots/' + planetname + '/11contour' + '_' + planetname + '.png', dpi=dpi)



    #saveaxs[1][4].set_title('Distribution of pressure profiles, log scale' + plottitlestr)
    log_grid = results['distr grid pressure']
    logdistr = saveaxs[1][4].imshow((np.transpose(log_grid)), cmap = 'plasma', norm = col.LogNorm(), aspect = 'auto', origin = 'lower', extent = (-0.5, N-0.5, -0.5, 2e12-0.5))
    cbar = savefig[1][4].colorbar(logdistr, ax=saveaxs[1][4])
    cbar.set_label(label = 'Relative frequency', labelpad = 14)
    saveaxs[1][4].invert_xaxis()
    saveaxs[1][4].set_xticks(xticklocations, xticklabels)
    saveaxs[1][4].set_xlabel("Normalised radius $r/R$")
    saveaxs[1][4].set_ylim(top = 2e12)
    saveaxs[1][4].locator_params(axis='y', nbins=5)
    #saveaxs[1][4].set_yticks(yticklocations, yticklabels)
    saveaxs[1][4].set_ylabel(r"$p$ [Pa]")
    #axins = zoomed_inset_axes(saveaxs[1][4], zoom = 2, loc='upper right')
    axins = inset_zoom_axes(saveaxs[1][4], [0.5, 0.5, 0.4, 0.4])
    #axins.imshow((np.transpose(log_grid)), cmap = 'plasma', norm = col.LogNorm(), aspect = 'auto', origin = 'lower', extent = (-0.5, N-0.5, -0.5, 2e4-0.5))
    axins.set_xticks([0,100,200,300], [1.0,0.9,0.8,0.7])
    axins.set_xlim(300,0)
    axins.set_ylim(0,1e11)
    #axins.set_yticks([0, 1000, 2000], [0, 1, 2])

    savefig[1][4].savefig('plots/' + planetname + '/14press_distr' + '_' + planetname + '.png', dpi=2*dpi)


    #saveaxs[2][0].set_title('Samples of the density profile distribution' + plottitlestr)
    #saveaxs[2][1].set_title('Samples of the density profile distribution' + plottitlestr + '\nFirst derivative, smoothed')
    #saveaxs[2][2].set_title('Samples of the density profile distribution' + plottitlestr + '\nFirst derivative, smoothed and normalised')
    ##saveaxs[2,2].set_title('Distribution Development Samples\nsecond derivative')
    n=59
    x = np.linspace(-3, 3, n)
    smoothing_kernel = np.exp(-0.5*x*x)
    smoothing_kernel /= np.sum(smoothing_kernel)
    #smoothing_kernel = np.array([1])
    #print(smoothing_kernel)
    #smoothing_kernel = np.array([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])/45
    nsamples = 15
    samplearray = np.linspace(23, 1023, nsamples, dtype = np.int32)
    samples = results['distr grid density'][samplearray, :]
    dev_colours = plt.get_cmap(name='jet')(np.linspace(0,1,nsamples))
    i=0
    for sample in samples:
        averaged_sample = np.convolve(sample, smoothing_kernel, mode = 'same')
        averaged_first_derivative = np.convolve(np.diff(averaged_sample), smoothing_kernel, mode = 'same')
        averaged_second_derivative = np.convolve(np.diff(averaged_first_derivative), smoothing_kernel, mode = 'same')
        #normalise to 1
        averaged_first_derivative_norm = averaged_first_derivative/np.max(np.abs(averaged_first_derivative))
        #averaged_second_derivative /= np.max(np.abs(averaged_second_derivative))
        saveaxs[2][0].semilogy(np.linspace(0, 2e4, RES), averaged_sample, color = dev_colours[i])
        saveaxs[2][1].plot(np.linspace(0, 2e4, RES-1), averaged_first_derivative, color = dev_colours[i])
        saveaxs[2][2].plot(np.linspace(0, 2e4, RES-1), averaged_first_derivative_norm, color = dev_colours[i])
        #saveaxs[2,2].plot(np.linspace(0, 2e4, RES-2), averaged_second_derivative, color = dev_colours[i])
        i+=1
        if i == 10:
            n=159
            x = np.linspace(-3, 3, n)
            smoothing_kernel = np.exp(-0.5*x*x)
            smoothing_kernel /= np.sum(smoothing_kernel)
    limit = 8000
    xticklocations_ = np.arange(0,limit+1,1000)
    xticklabels_ = np.arange(0,limit//1000+1,1)
    saveaxs[2][0].set_ylim(bottom = 1e-4)
    saveaxs[2][0].set_xlim(left = 0, right = limit)
    saveaxs[2][0].set_xticks(xticklocations_, xticklabels_)
    saveaxs[2][0].set_xlabel(r"$\rho$ [1000 kg m$^{-3}$]")
    saveaxs[2][0].set_ylabel("Relative frequency")
    saveaxs[2][1].set_yscale('symlog')
    #saveaxs[2][1].yaxis.set_major_formatter('{x:.0e}')
    #saveaxs[2][1].get_yaxis().set_ticks([])
    saveaxs[2][1].set_ylim(bottom = -0.5e-3, top = 0.5e-3)
    saveaxs[2][1].set_xlim(left = 0, right = limit)
    saveaxs[2][1].set_xticks(xticklocations_, xticklabels_)
    saveaxs[2][1].set_xlabel(r"$\rho$ [1000 kg m$^{-3}$]")
    saveaxs[2][1].set_ylabel("Change in frequency", labelpad = -30)
    saveaxs[2][2].set_yscale('symlog')
    saveaxs[2][2].get_yaxis().set_ticks([])
    #saveaxs[2][2].set_ylim(bottom = -3e-4, top = 3e-4)
    saveaxs[2][2].set_xlim(left = 0, right = limit)
    saveaxs[2][2].set_xticks(xticklocations_, xticklabels_)
    saveaxs[2][2].set_xlabel(r"$\rho$ [1000 kg m$^{-3}$]")
    saveaxs[2][2].set_ylabel("Change in frequency (normalised)")

    sampleradii = 1 - samplearray/1023
    sampleradiistrings = ['%.2f' % x for x in sampleradii]

    cbar = savefig[2][0].colorbar(cm.ScalarMappable(norm=col.NoNorm(), cmap = 'jet'), ax=saveaxs[2][0], ticks = np.linspace(0,1,nsamples), boundaries = np.linspace(0 - 0.5/nsamples, 1 + 0.5/nsamples,nsamples + 1), values = np.linspace(0,1,nsamples))
    cbar.ax.set_yticklabels(sampleradiistrings)
    cbar.set_label(label = r'Sample radius $r/R$')

    cbar = savefig[2][1].colorbar(cm.ScalarMappable(norm=col.NoNorm(), cmap = 'jet'), ax=saveaxs[2][1], ticks = np.linspace(0,1,nsamples), boundaries = np.linspace(0 - 0.5/nsamples, 1 + 0.5/nsamples,nsamples + 1), values = np.linspace(0,1,nsamples))
    cbar.ax.set_yticklabels(sampleradiistrings)
    cbar.set_label(label = r'Sample radius $r/R$')

    cbar = savefig[2][2].colorbar(cm.ScalarMappable(norm=col.NoNorm(), cmap = 'jet'), ax=saveaxs[2][2], ticks = np.linspace(0,1,nsamples), boundaries = np.linspace(0 - 0.5/nsamples, 1 + 0.5/nsamples,nsamples + 1), values = np.linspace(0,1,nsamples))
    cbar.ax.set_yticklabels(sampleradiistrings)
    cbar.set_label(label = r'Sample radius $r/R$')

    savefig[2][0].savefig('plots/' + planetname + '/20dev' + '_' + planetname + '.png', dpi=dpi)
    savefig[2][1].savefig('plots/' + planetname + '/21devder' + '_' + planetname + '.png', dpi=dpi)
    savefig[2][2].savefig('plots/' + planetname + '/22devdernorm' + '_' + planetname + '.png', dpi=dpi)

    #do it all again but for a subsample

    #saveaxs[4][0].set_title('Samples of a section of the distribution' + plottitlestr)
    #saveaxs[4][1].set_title('Samples of a section of the distribution' + plottitlestr + '\nFirst derivative, smoothed')
    #saveaxs[4][2].set_title('Samples of a section of the distribution' + plottitlestr + '\nFirst derivative, smoothed and normalised')
    ##saveaxs[2,2].set_title('Distribution Development Samples\nsecond derivative')
    n=59
    x = np.linspace(-3, 3, n)
    smoothing_kernel = np.exp(-0.5*x*x)
    smoothing_kernel /= np.sum(smoothing_kernel)
    #smoothing_kernel = np.array([1])
    #print(smoothing_kernel)
    #smoothing_kernel = np.array([1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])/45
    nsamples = 12
    samplearray = np.linspace(150, 650, nsamples, dtype = np.int32)
    samples = results['distr grid density'][samplearray, :]
    dev_colours = plt.get_cmap(name='jet')(np.linspace(0,1,nsamples))
    i=0
    for sample in samples:
        averaged_sample = np.convolve(sample, smoothing_kernel, mode = 'same')
        averaged_first_derivative = np.convolve(np.diff(averaged_sample), smoothing_kernel, mode = 'same')
        averaged_second_derivative = np.convolve(np.diff(averaged_first_derivative), smoothing_kernel, mode = 'same')
        #normalise to 1
        averaged_first_derivative_norm = averaged_first_derivative/np.max(np.abs(averaged_first_derivative))
        #averaged_second_derivative /= np.max(np.abs(averaged_second_derivative))
        saveaxs[4][0].semilogy(np.linspace(0, 2e4, RES), averaged_sample, color = dev_colours[i])
        saveaxs[4][1].plot(np.linspace(0, 2e4, RES-1), averaged_first_derivative, color = dev_colours[i])
        saveaxs[4][2].plot(np.linspace(0, 2e4, RES-1), averaged_first_derivative_norm, color = dev_colours[i])
        #saveaxs[2,2].plot(np.linspace(0, 2e4, RES-2), averaged_second_derivative, color = dev_colours[i])
        i+=1

    limit = 5000
    xticklocations_ = np.arange(0,limit+1,1000)
    xticklabels_ = np.arange(0,limit//1000+1,1)
    saveaxs[4][0].set_ylim(bottom = 1e-4)
    saveaxs[4][0].set_xlim(left = 0, right = limit)
    saveaxs[4][0].set_xticks(xticklocations_, xticklabels_)
    saveaxs[4][0].set_xlabel(r"$\rho$ [1000 kg m$^{-3}$]")
    saveaxs[4][0].set_ylabel("Relative frequency")
    saveaxs[4][1].set_yscale('symlog')
    #saveaxs[2][1].yaxis.set_major_formatter('{x:.0e}')
    #saveaxs[2][1].get_yaxis().set_ticks([])
    saveaxs[4][1].set_ylim(bottom = -0.5e-3, top = 0.5e-3)
    saveaxs[4][1].set_xlim(left = 0, right = limit)
    saveaxs[4][1].set_xticks(xticklocations_, xticklabels_)
    saveaxs[4][1].set_xlabel(r"$\rho$ [1000 kg m$^{-3}$]")
    saveaxs[4][1].set_ylabel("Change in frequency", labelpad = -30)
    saveaxs[4][2].set_yscale('symlog')
    saveaxs[4][2].get_yaxis().set_ticks([])
    #saveaxs[2][2].set_ylim(bottom = -3e-4, top = 3e-4)
    saveaxs[4][2].set_xlim(left = 0, right = limit)
    saveaxs[4][2].set_xticks(xticklocations_, xticklabels_)
    saveaxs[4][2].set_xlabel(r"$\rho$ [1000 kg m$^{-3}$]")
    saveaxs[4][2].set_ylabel("Change in frequency (normalised)")

    sampleradii = 1 - samplearray/1023
    sampleradiistrings = ['%.2f' % x for x in sampleradii]

    cbar = savefig[4][0].colorbar(cm.ScalarMappable(norm=col.NoNorm(), cmap = 'jet'), ax=saveaxs[4][0], ticks = np.linspace(0,1,nsamples), boundaries = np.linspace(0 - 0.5/nsamples, 1 + 0.5/nsamples,nsamples + 1), values = np.linspace(0,1,nsamples))
    cbar.ax.set_yticklabels(sampleradiistrings)
    cbar.set_label(label = r'Sample radius $r/R$')

    cbar = savefig[4][1].colorbar(cm.ScalarMappable(norm=col.NoNorm(), cmap = 'jet'), ax=saveaxs[4][1], ticks = np.linspace(0,1,nsamples), boundaries = np.linspace(0 - 0.5/nsamples, 1 + 0.5/nsamples,nsamples + 1), values = np.linspace(0,1,nsamples))
    cbar.ax.set_yticklabels(sampleradiistrings)
    cbar.set_label(label = r'Sample radius $r/R$')

    cbar = savefig[4][2].colorbar(cm.ScalarMappable(norm=col.NoNorm(), cmap = 'jet'), ax=saveaxs[4][2], ticks = np.linspace(0,1,nsamples), boundaries = np.linspace(0 - 0.5/nsamples, 1 + 0.5/nsamples,nsamples + 1), values = np.linspace(0,1,nsamples))
    cbar.ax.set_yticklabels(sampleradiistrings)
    cbar.set_label(label = r'Sample radius $r/R$')

    savefig[4][0].savefig('plots/' + planetname + '/40dev' + '_' + planetname + '.png', dpi=dpi)
    savefig[4][1].savefig('plots/' + planetname + '/41devder' + '_' + planetname + '.png', dpi=dpi)
    savefig[4][2].savefig('plots/' + planetname + '/42devdernorm' + '_' + planetname + '.png', dpi=dpi)





    #thiccness

    #saveaxs[2][3].set_title('Size of confidence intervals' + plottitlestr)
    onesigmathiccness = uppersd - lowersd
    twosigmathiccness = twouppersd - twolowersd
    #saveaxs[2,3].plot(onesigmathiccness, color = 'black', linewidth = 1)
    #saveaxs[2,3].plot(twosigmathiccness, color = 'black', linewidth = 1)
    saveaxs[2][3].fill_between(np.arange(N), 0                , onesigmathiccness, color = '#fca636', label = '68%')
    saveaxs[2][3].fill_between(np.arange(N), onesigmathiccness, twosigmathiccness, color = '#6a006c', label = '95%')
    saveaxs[2][3].invert_xaxis()
    saveaxs[2][3].set_yticks(yticklocations, yticklabels)
    saveaxs[2][3].set_xticks(xticklocations, xticklabels)
    saveaxs[2][3].set_xlabel("Normalised radius $r/R$")
    saveaxs[2][3].set_ylabel(r"$\Delta\rho$ [1000 kg m$^{-3}$]")
    saveaxs[2][3].legend()

    savefig[2][3].savefig('plots/' + planetname + '/23confintsize' + '_' + planetname + '.png', dpi=dpi)

    #minmax dens & changes
    #this is 110% sbeve certified mcterrible™
    """
    #saveaxs[3][0].set_title('Min Density of J explained\nmode (p), median (o), average (r)')
    mindensbinvalues, mindensbinedges, _  = saveaxs[3][0].hist(results['min dens'][results['dens Js explained view']], bins = bincount)
    mindensbinlocations = (mindensbinedges[:-1]+mindensbinedges[1:])/2
    modeindex = np.argmax(mindensbinvalues)
    modelocation = mindensbinlocations[modeindex]
    modevalue = mindensbinvalues[modeindex]
    weighted_indices = mindensbinvalues*np.arange(len(mindensbinvalues))
    avgindex = int(np.sum(weighted_indices)/np.sum(mindensbinvalues))
    avglocation = mindensbinlocations[avgindex]
    avgvalue = mindensbinvalues[avgindex]
    median = np.median(results['min dens'][results['dens Js explained view']])
    medianindex = np.digitize(median, mindensbinedges)
    medianlocation = mindensbinlocations[medianindex]
    medianvalue = mindensbinvalues[medianindex]
    
    saveaxs[3][0].vlines(modelocation, ymin = 0, ymax = modevalue, linestyle ='--', color = 'tab:purple')
    saveaxs[3][0].text(modelocation, modevalue, '{:.0f}'.format(modelocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', backgroundcolor='white', fontsize = 'small')
    saveaxs[3][0].vlines(avglocation, ymin = 0, ymax = avgvalue, linestyle ='--', color = 'tab:red')
    saveaxs[3][0].text(avglocation, avgvalue, '{:.0f}'.format(avglocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', backgroundcolor='white', fontsize = 'small')
    saveaxs[3][0].vlines(medianlocation, ymin = 0, ymax = medianvalue, linestyle ='--', color = 'tab:orange')
    saveaxs[3][0].text(medianlocation, medianvalue, '{:.0f}'.format(medianlocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', backgroundcolor='white', fontsize = 'small')

    savefig[3][0].savefig('plots/' + planetname + '/30mindens' + '_' + planetname + '.png', dpi=dpi)
    """


    #saveaxs[3][1].set_title('Core Density' + plottitlestr)
    maxdensbinvalues, maxdensbinedges, _  = saveaxs[3][1].hist(results['max dens'][results['dens Js explained view']], bins = 400, color = defaultcolor, density = True)
    maxdensbinlocations = (maxdensbinedges[:-1]+maxdensbinedges[1:])/2
    
    modeindex = np.argmax(maxdensbinvalues)
    modelocation = maxdensbinlocations[modeindex]
    modevalue = maxdensbinvalues[modeindex]

    weighted_indices = maxdensbinvalues*np.arange(len(maxdensbinvalues))
    avgindex = int(np.sum(weighted_indices)/np.sum(maxdensbinvalues))
    avglocation = maxdensbinlocations[avgindex]
    avgvalue = maxdensbinvalues[avgindex]

    median = np.median(results['max dens'][results['dens Js explained view']])
    medianindex = np.digitize(median, maxdensbinedges)
    medianlocation = maxdensbinlocations[medianindex]
    medianvalue = maxdensbinvalues[medianindex]


    lowersdindex = np.digitize(lowersd[-1], maxdensbinedges)
    lowersdlocation = maxdensbinlocations[lowersdindex]
    lowersdvalue = maxdensbinvalues[lowersdindex]

    uppersdindex = np.digitize(uppersd[-1], maxdensbinedges)
    uppersdlocation = maxdensbinlocations[uppersdindex]
    uppersdvalue = maxdensbinvalues[uppersdindex]

    #twolowersdindex = np.digitize(twolowersd[-1], maxdensbinedges)
    #twolowersdlocation = maxdensbinlocations[twolowersdindex]
    #twolowersdvalue = maxdensbinvalues[twolowersdindex]

    #twouppersdindex = np.digitize(twouppersd[-1], maxdensbinedges)
    #twouppersdlocation = maxdensbinlocations[twouppersdindex]
    #twouppersdvalue = maxdensbinvalues[twouppersdindex]

    #saveaxs[3][1].vlines(modelocation, ymin = 0, ymax = modevalue, linestyle ='-', color = 'xkcd:violet')
    #saveaxs[3][1].text(modelocation, modevalue, '{:.2f}'.format(modelocation/1000), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')

    #saveaxs[3][1].vlines(twolowersdlocation, ymin = 0, ymax = twolowersdvalue, linestyle = (0, (1, 1)), color = 'black', label = '2nd percentile')
    #saveaxs[3][1].text(twolowersdlocation, twolowersdvalue, '{:.2f}'.format(twolowersdlocation/1000), ha='right', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')
    saveaxs[3][1].vlines(lowersdlocation, ymin = 0, ymax = lowersdvalue, linestyle ='--', color = 'black')#, label = '16th percentile')
    saveaxs[3][1].text(lowersdlocation, lowersdvalue, '{:.2f}'.format(lowersdlocation/1000), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')
    saveaxs[3][1].vlines(medianlocation, ymin = 0, ymax = medianvalue, linestyle ='-', color = 'black', label = 'Median')
    saveaxs[3][1].text(medianlocation, medianvalue, '{:.2f}'.format(medianlocation/1000), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')
    saveaxs[3][1].vlines(uppersdlocation, ymin = 0, ymax = uppersdvalue, linestyle ='--', color = 'black', label = r'$\pm\sigma$')
    saveaxs[3][1].text(uppersdlocation, uppersdvalue, '{:.2f}'.format(uppersdlocation/1000), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')
    #saveaxs[3][1].vlines(twouppersdlocation, ymin = 0, ymax = twouppersdvalue, linestyle = (0, (1, 1)), color = 'black', label = '98th percentile')
    #saveaxs[3][1].text(twouppersdlocation, twouppersdvalue, '{:.2f}'.format(twouppersdlocation/1000), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')

    saveaxs[3][1].vlines(avglocation, ymin = 0, ymax = avgvalue, linestyle ='-', color = 'tab:red', label = 'Average')
    saveaxs[3][1].text(avglocation, avgvalue, '{:.2f}'.format(avglocation/1000), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')

    saveaxs[3][1].set_ylabel('Relative frequency')
    saveaxs[3][1].get_yaxis().set_ticks([])
    saveaxs[3][1].set_xticks(yticklocations, yticklabels)
    saveaxs[3][1].set_xlabel(r"$\rho$ [1000 kg m$^{-3}$]")

    saveaxs[3][1].legend()

    savefig[3][1].savefig('plots/' + planetname + '/31maxdens' + '_' + planetname + '.png', dpi=dpi)

    #saveaxs[3][2].set_title('Average of profiles' + plottitlestr)
    saveaxs[3][2].plot(results['avg start'], linestyle ='--', color = 'tab:gray', label = 'Generated')
    saveaxs[3][2].plot(results['avg respected start'], color = 'tab:purple', label = r'$\rho_{\text{max}}$ respected')
    saveaxs[3][2].plot(results['avg successful start'], color = 'tab:red', label = 'Successful start')
    saveaxs[3][2].plot(results['avg successful result'], color = 'tab:blue', label = 'Successful result')
    saveaxs[3][2].invert_xaxis()
    saveaxs[3][2].set_xticks(xticklocations, xticklabels)
    saveaxs[3][2].set_xlabel("Normalised radius $r/R$")
    saveaxs[3][2].set_yticks(yticklocations, yticklabels)
    saveaxs[3][2].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")

    saveaxs[3][2].legend()

    savefig[3][2].savefig('plots/' + planetname + '/32avgstartrespsuccres' + '_' + planetname + '.png', dpi=dpi)


    #saveaxs[3][3].set_title('Average change' + plottitlestr)
    posneg = ['tab:blue' if y >= 0 else 'tab:red' for y in results['avg change']]
    saveaxs[3][3].bar(range(1024), results['avg change'], width=1, bottom = 0, color = posneg)
    saveaxs[3][3].axhline(0, linestyle ='--', color = 'tab:gray')
    saveaxs[3][3].invert_xaxis()
    saveaxs[3][3].set_xticks(xticklocations, xticklabels)
    saveaxs[3][3].set_xlabel("Normalised radius $r/R$")
    saveaxs[3][3].set_ylabel(r"$\Delta\rho$ [kg m$^{-3}$]")

    savefig[3][3].savefig('plots/' + planetname + '/33avgchange' + '_' + planetname + '.png', dpi=dpi)


    #NMoI
    #saveaxs[4][3].set_title('Normalised Moments of Inertia' + plottitlestr)
    maxnmoibinvalues, maxnmoibinedges, _  = saveaxs[4][3].hist(results['moments of inertia'], bins = 200, color = defaultcolor, density = True)
    maxnmoibinlocations = (maxnmoibinedges[:-1]+maxnmoibinedges[1:])/2
    
    weighted_indices = maxnmoibinvalues*np.arange(len(maxnmoibinvalues))
    avgindex = int(np.sum(weighted_indices)/np.sum(maxnmoibinvalues))
    avglocation = maxnmoibinlocations[avgindex]
    avgvalue = maxnmoibinvalues[avgindex]
    """
    median = np.median(results['moments of inertia'])
    medianindex = np.digitize(median, maxnmoibinedges)
    medianlocation = maxnmoibinlocations[medianindex]
    medianvalue = maxnmoibinvalues[medianindex]
    """

    saveaxs[4][3].vlines(avglocation, ymin = 0, ymax = avgvalue, linestyle ='-', color = 'tab:red', label = 'Average')
    saveaxs[4][3].text(avglocation, avgvalue, '{:.6f}'.format(avglocation), ha='right', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small', backgroundcolor = 'white')

    #saveaxs[4][3].vlines(medianlocation, ymin = 0, ymax = medianvalue, linestyle ='-', color = 'black', label = 'Median')
    #saveaxs[4][3].text(medianlocation, medianvalue, '{:.2f}'.format(medianlocation/1000), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')

    saveaxs[4][3].set_ylabel('Relative frequency')
    #saveaxs[4][3].set_xticks(yticklocations, yticklabels)
    saveaxs[4][3].set_xlabel(r"NMoI [unitless]")

    saveaxs[4][3].legend()

    saveaxs[4][3].get_yaxis().set_ticks([])

    savefig[4][3].savefig('plots/' + planetname + '/43nmoi' + '_' + planetname + '.png', dpi=dpi)





    #FR
    #saveaxs[4][4].set_title('Flattening Ratios' + plottitlestr)
    maxfrbinvalues, maxfrbinedges, _  = saveaxs[4][4].hist(results['flattening ratios'], bins = 200, color = defaultcolor, density = True)
    maxfrbinlocations = (maxfrbinedges[:-1]+maxfrbinedges[1:])/2
    
    weighted_indices = maxfrbinvalues*np.arange(len(maxfrbinvalues))
    avgindex = int(np.sum(weighted_indices)/np.sum(maxfrbinvalues))
    avglocation = maxfrbinlocations[avgindex]
    avgvalue = maxfrbinvalues[avgindex]
    """
    median = np.median(results['flattening ratios'])
    medianindex = np.digitize(median, maxfrbinedges)
    medianlocation = maxfrbinlocations[medianindex]
    medianvalue = maxfrbinvalues[medianindex]
    """
    saveaxs[4][4].vlines(avglocation, ymin = 0, ymax = avgvalue, linestyle ='-', color = 'tab:red', label = 'Average')
    saveaxs[4][4].text(avglocation, avgvalue, '{:.8f}'.format(avglocation), ha='right', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small', backgroundcolor = 'white')

    #saveaxs[4][4].vlines(medianlocation, ymin = 0, ymax = medianvalue, linestyle ='-', color = 'black', label = 'Median')
    #saveaxs[4][4].text(medianlocation, medianvalue, '{:.2f}'.format(medianlocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')

    saveaxs[4][4].set_ylabel('Relative frequency')
    #saveaxs[4][4].set_xticks(yticklocations, yticklabels)
    saveaxs[4][4].set_xlabel(r"$R_{mean}/R_{eq}$ [unitless]")

    saveaxs[4][4].legend()

    saveaxs[4][4].get_yaxis().set_ticks([])

    savefig[4][4].savefig('plots/' + planetname + '/44fr' + '_' + planetname + '.png', dpi=dpi)




    """
    #3d plot
    x = np.arange(N)
    y = np.linspace(0, 2e4, RES)
    X, Y = np.meshgrid(x, y)
    Z = np.transpose(results['distr grid density'])
    threedeefig, threedeeax = plt.subplots(subplot_kw = dict(projection = '3d'))
 #   threedeeax.set_title('3d Distribution')
    surf = threedeeax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='plasma', linewidth=0, edgecolor='none')
    """
    print()
    print("Number of datasets analysed:     " + str(results['n']))
    print("Number of which respect rho_MAX: " + str(results['nR']))
    print("Number of which explain Js:      " + str(results['nJ']))
    print("Percentage respect rho_MAX:      " + '{:.2%}'.format(results['rho_MAX respected%']))
    print("Percentage explained Js:         " + '{:.2%}'.format(results['Js explained%']))
    print("Time to generate plots:          " + time.strftime("%M:%S", time.gmtime((time.perf_counter()-toc))))
    #fig.show()
    return

generate_plots()
