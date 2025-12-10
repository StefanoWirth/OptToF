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
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

def _default_mpl_opts():

    """
    Return default Matplotlib rcParams options for all plots.

    These settings control visual aspects such as line width, marker size,
    tick style, font size, and figure dimensions. They serve as a base for
    all plotting functions in this module.

    Returns
    -------
    dict
        Dictionary of Matplotlib rcParams keys and default values.
    """

    opts = {}

    #Lines and errobar: 
    opts['lines.linewidth']             = 3.0
    opts['lines.markersize']            = 9
    opts['errorbar.capsize']            = 0.0

    #Axes and margins:
    opts['axes.labelsize']              = 18
    opts['axes.xmargin']                = 0.0
    opts['axes.ymargin']                = 0.0
    opts['axes.formatter.useoffset']    = False

    #Ticks:
    opts['xtick.labelsize']             = 16
    opts['ytick.labelsize']             = 16
    opts['xtick.direction']             = 'in'
    opts['ytick.direction']             = 'in'
    opts['xtick.top']                   = True
    opts['ytick.right']                 = True
    opts['xtick.minor.visible']         = True
    opts['ytick.minor.visible']         = True
    opts['xtick.major.size']            = 7.0
    opts['ytick.major.size']            = 7.0
    opts['xtick.minor.size']            = 4.0
    opts['ytick.minor.size']            = 4.0
    opts['xtick.major.width']           = 1.6
    opts['ytick.major.width']           = 1.6
    opts['xtick.major.pad']             = 7.0
    opts['ytick.major.pad']             = 7.0

    #Legend:
    opts['legend.title_fontsize']       = 16
    opts['legend.fontsize']             = 16
    
    #Figure properties:
    opts['figure.figsize']              = [6.4, 4.8]
    opts['figure.dpi']                  = 200
    opts['font.family']                 = 'Ubuntu'

    return opts

def _apply_mpl_opts(opts):

    """
    Apply Matplotlib rcParams options from a dictionary.

    Parameters
    ----------
    opts : dict
        Dictionary of rcParams-style key-value pairs to apply globally.
    """

    for kwd, value in opts.items():

        try:

            plt.rcParams[kwd] = value

        except:

            continue

opts = _default_mpl_opts(); _apply_mpl_opts(opts)

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

rho_max = 2e4
p_max = 3e12

is_neptune = False
dodistcomparison = True

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
        with open('result_dict_uranus_corr_dist_new_all_fixed.pkl', 'rb') as f:
            results = pickle.load(f)


    toc = time.perf_counter()



    #PLOT ================================

    fig, axs = plt.subplots(6, 6, layout = 'constrained', num = str(is_neptune))

    if is_neptune == True:
        planetname = 'neptune'
        plottitlestr = 'Neptune'
        defaultcolor = '#3b55e1'
    else:
        planetname = 'uranus_corr'
        plottitlestr = 'Uranus'
        defaultcolor = '#50b5ad'

    savefig = [[None for i in range(7)] for j in range(7)]
    saveaxs = [[None for i in range(7)] for j in range(7)]

    for i in range(7):
        for j in range(7):
            savefig[i][j], saveaxs[i][j] = plt.subplots(num = str([i,j]))

    dpi = 200

    bincount = N    #THE BINS MARTY THE BINS WHAT DO THEY MEAN

    weights = results["weights"]

    nticks = 5

    xticklocations = np.linspace(1023, 0, nticks)
    xticklabels = np.linspace(0, 1, nticks)
    yticklocations = [0, 5000, 10000, 15000, 20000]
    yticklabels = [0, 5, 10, 15, 20]


    if dodistcomparison:
        # Start Result Size comparison
        #running_average_starting_rhos = results['average starting rhos']
        running_average_starting_rhos_dist = results['average starting rhos dist']
        #running_average_rhos = results['average rhos']
        running_average_rhos_dist = results['average rhos dist']
        n_close_rhos = results['n close rhos']

        cutoff = 1
        """
        for i in range(5):
            if i != 3: continue
            saveaxs[6][0].hist(running_average_starting_rhos_dist[i, n_close_rhos[i] > cutoff], bins = 20, alpha = 0.5)
        savefig[6][0].savefig('plots/' + planetname + '/60running_average_starting_rhos_dist' + '_' + planetname + '.png', dpi=dpi)
        
        for i in range(5):
            if i != 3: continue
            saveaxs[6][1].hist(running_average_rhos_dist[i, n_close_rhos[i] > cutoff], bins = 20, alpha = 0.5)
        savefig[6][1].savefig('plots/' + planetname + '/61running_average_rhos_dist' + '_' + planetname + '.png', dpi=dpi)
        
        for i in range(5):
            if i != 3: continue
            saveaxs[6][2].hist(n_close_rhos[i, n_close_rhos[i] > cutoff], bins = 20, alpha = 0.5)
            print(f"{i}. Discarded bois: {np.size(n_close_rhos[i, n_close_rhos[i] <= cutoff])}")
        savefig[6][2].savefig('plots/' + planetname + '/62n_close_rhos' + '_' + planetname + '.png', dpi=dpi)
        
        for i in range(5):
            if i != 3: continue
            saveaxs[6][3].scatter(running_average_rhos_dist[i, n_close_rhos[i] > cutoff], running_average_starting_rhos_dist[i, n_close_rhos[i] > cutoff], s=0.5)
        #saveaxs[6][3].set_xlim(left = 0, right = 7e7)
        #saveaxs[6][3].set_ylim(bottom = 0, top = 70*7e7)
        #saveaxs[6][3].set_yscale("log")
        savefig[6][3].savefig('plots/' + planetname + '/63distcomp' + '_' + planetname + '.png', dpi=dpi)
        """
        size_ratios = [running_average_starting_rhos_dist[i, n_close_rhos[i] > cutoff]/running_average_rhos_dist[i, n_close_rhos[i] > cutoff]/np.mean(running_average_starting_rhos_dist[i, n_close_rhos[i] > cutoff]/running_average_rhos_dist[i, n_close_rhos[i] > cutoff]) for i in range(5)]

        i = 2
        saveaxs[6][4].hist(size_ratios[i], bins = 16, range = (0,4))
        print(f"2 Lower sigma (2.5th  percentile): {np.percentile(size_ratios[i], q=2.5)}")
        print(f"  Lower sigma (16th   percentile): {np.percentile(size_ratios[i], q=16)}")
        print(f"  Upper sigma (84th   percentile): {np.percentile(size_ratios[i], q=84)}")
        print(f"2 Upper sigma (97.5th percentile): {np.percentile(size_ratios[i], q=97.5)}")
        saveaxs[6][4].set_xlabel(r"Discoverability $\mathfrak{d}$ [unitless]")
        saveaxs[6][4].set_ylabel("Number of reference solutions")
        savefig[6][4].savefig('plots/' + planetname + '/64sizeratios' + '_' + planetname + '.png', dpi=dpi)
        
        saveaxs[6][3].hist(np.emath.logn(4,size_ratios[i]), bins = 16, range = (-1,1))
        print(f"2 Lower sigma (2.5th  percentile): {np.percentile(np.emath.logn(4,size_ratios[i]), q=2.5)}")
        print(f"  Lower sigma (16th   percentile): {np.percentile(np.emath.logn(4,size_ratios[i]), q=16)}")
        print(f"  Upper sigma (84th   percentile): {np.percentile(np.emath.logn(4,size_ratios[i]), q=84)}")
        print(f"2 Upper sigma (97.5th percentile): {np.percentile(np.emath.logn(4,size_ratios[i]), q=97.5)}")
        saveaxs[6][3].set_xlabel("Error")
        saveaxs[6][3].set_ylabel("Number of reference solutions")
        savefig[6][3].savefig('plots/' + planetname + '/63error' + '_' + planetname + '.png', dpi=dpi)
        

        saveaxs[6][5].scatter(n_close_rhos[i, n_close_rhos[i] > cutoff], size_ratios[i], s=5)
        saveaxs[6][5].set_ylim(bottom = 0, top = 4)
        saveaxs[6][5].set_ylabel(r"Discoverability $\mathfrak{d}$ [unitless]")
        saveaxs[6][5].set_xlabel("Popularity (Number of neighbours)")
        savefig[6][5].savefig('plots/' + planetname + '/65popularityvsratio' + '_' + planetname + '.png', dpi=dpi)

        saveaxs[6][6].scatter(n_close_rhos[i, n_close_rhos[i] > cutoff], np.emath.logn(4,size_ratios[i]), s=5)
        saveaxs[6][6].set_ylim(bottom = -1, top = 1)
        saveaxs[6][6].set_ylabel("Error")
        saveaxs[6][6].set_xlabel("Popularity (Number of neighbours)")
        #saveaxs[6][6].set_xscale("log")
        #saveaxs[6][6].set_yscale("log")
        savefig[6][6].savefig('plots/' + planetname + '/66logpopularityvslogratio' + '_' + planetname + '.png', dpi=dpi)
        
        max_distances = [3e2, 4e2, 5e2, 6e2, 7e2]
        bins_x = np.logspace(-4, 4, 33, base=2)
        import matplotlib
        legend_patches=[]
        for i in range(5):
            counts, bin_edges = np.histogram(size_ratios[i], bins=bins_x)
            x_outline_x = np.repeat(bin_edges, 2)
            y_outline_x = np.hstack(([0], np.repeat(counts, 2), [0]))
            saveaxs[3][6].plot(x_outline_x, y_outline_x, alpha = 1, label=f"{max_distances[i]:.0f}")
            saveaxs[3][6].hist(size_ratios[i], bins = bins_x, alpha = 0.2, color="C{}".format(i))#, label=f"Max Dist {max_distances[i]}")
            hass = np.array(matplotlib.colors.to_rgba('C'+str(i)))
            hass[3]=0.2
            legend_patches.append(matplotlib.patches.Patch(facecolor=hass, edgecolor='C'+str(i), linewidth=3.0, label=f"{max_distances[i]:.0f}"))
        
        saveaxs[3][6].legend(handles=legend_patches,title=r"$\epsilon/2$ [kg m$^{-3}$]", bbox_to_anchor=(0.0, 0.0, 0.95, 0.95))
        saveaxs[3][6].set_xscale("log", base=2)
        saveaxs[3][6].set_xlabel(r"Discoverability $\mathfrak{d}$ [unitless]")
        saveaxs[3][6].set_ylabel("Number of reference solutions")
        savefig[3][6].savefig('plots/' + planetname + '/36errors_all' + '_' + planetname + '.png', dpi=dpi)
        """
        bins_x = np.linspace(0, 4, 17)
        for i in range(5):
            counts, bin_edges = np.histogram(size_ratios[i], bins=bins_x)
            x_outline_x = np.repeat(bin_edges, 2)
            y_outline_x = np.hstack(([0], np.repeat(counts, 2), [0]))
            saveaxs[4][6].plot(x_outline_x, y_outline_x, alpha = 0.6, label=f"{max_distances[i]:.0f}")
            #saveaxs[4][6].hist(size_ratios[i], bins = 16, range = (0,4), alpha = 0.6, label=f"Max Dist {max_distances[i]}")
        savefig[4][6].legend()
        savefig[4][6].savefig('plots/' + planetname + '/46sizeratios_all' + '_' + planetname + '.png', dpi=dpi)
        """
        for i in range(5):
            saveaxs[5][6].scatter(n_close_rhos[i, n_close_rhos[i] > cutoff], size_ratios[i], s=5, label=f"{max_distances[i]:.0f}")
        saveaxs[5][6].set_ylim(bottom = 2**-4, top = 2**4)
        saveaxs[5][6].set_xscale("log")
        saveaxs[5][6].set_yscale("log", base=2)
        saveaxs[5][6].set_ylabel(r"Discoverability $\mathfrak{d}$ [unitless]")
        saveaxs[5][6].set_xlabel("Popularity (Number of neighbours)")
        #saveaxs[5][6].legend()
        savefig[5][6].savefig('plots/' + planetname + '/56popularityvsratio_all' + '_' + planetname + '.png', dpi=dpi)

        import math
        
        
        print("Individual")
        for i in range(5):
            print(f"{i+1}.")
            print(f"N Lost:  {np.size(n_close_rhos[i, n_close_rhos[i] <= cutoff])}")
            print(f"Range:   {np.percentile(size_ratios[i], q=2.5):.3f}-{np.percentile(size_ratios[i], q=97.5):.3f}")
            print(f"Span:    {np.percentile(size_ratios[i], q=97.5)/np.percentile(size_ratios[i], q=2.5):.2f}")
            linregress_result = linregress(n_close_rhos[i, n_close_rhos[i] > cutoff], size_ratios[i])
            #linregress_result = linregress(running_average_rhos_dist[i, n_close_rhos[i] > cutoff], running_average_starting_rhos_dist[i, n_close_rhos[i] > cutoff])
            print(f"Slope:   {linregress_result.slope}")
            print(f"R-value: {linregress_result.rvalue}")
            print(f"P-value: {linregress_result.pvalue}")
            print(f"Std-err: {linregress_result.stderr}")
            print()
            #reg = LinearRegression().fit(running_average_rhos_dist[i, n_close_rhos[i] > cutoff].reshape(-1, 1), running_average_starting_rhos_dist[i, n_close_rhos[i] > cutoff].reshape(-1, 1), n_close_rhos[i, n_close_rhos[i] > cutoff])
            #print(f"R-value: {math.sqrt(reg.score(running_average_rhos_dist[i, n_close_rhos[i] > cutoff].reshape(-1, 1), running_average_starting_rhos_dist[i, n_close_rhos[i] > cutoff].reshape(-1, 1), n_close_rhos[i, n_close_rhos[i] > cutoff]))}")
        
        """
        running_average_rhos_dist_all = np.concatenate([running_average_rhos_dist[i, n_close_rhos[i] > cutoff] for i in range(5)])
        running_average_starting_rhos_dist_all = np.concatenate([running_average_starting_rhos_dist[i, n_close_rhos[i] > cutoff] for i in range(5)])
        n_close_rhos_all = np.concatenate([n_close_rhos[i, n_close_rhos[i] > cutoff] for i in range(5)])
        ratios_all = np.concatenate([running_average_starting_rhos_dist[i, n_close_rhos[i] > cutoff]/running_average_rhos_dist[i, n_close_rhos[i] > cutoff] for i in range(5)])

        print("Big")
        linregress_result = linregress(n_close_rhos_all, ratios_all)
        #linregress_result = linregress(running_average_rhos_dist_all, running_average_starting_rhos_dist_all)
        
        print(f"Slope:   {linregress_result.slope}")
        print(f"R-value: {linregress_result.rvalue}")
        print(f"P-value: {linregress_result.pvalue}")
        print(f"Std-err: {linregress_result.stderr}")
        """


    # J2 vs J4 (outliers killed)

    # Data for histograms:
    bins = 120
    bins_hist = 15
    gamma = 1 #Gamma correction for better color visibility
    colors = ['C0', 'C1']
    colormaps = ['Blues', 'Oranges']

    kappa = 2

    #Create new figure:
    jfig = plt.figure(num='J2J4', figsize=[6.4, 6.4], layout='constrained')
    gs = jfig.add_gridspec(2, 2, width_ratios=(4, 1.25), height_ratios=(1.25, 4), wspace=0.00, hspace=0.00)

    #Define axes:
    ax          = jfig.add_subplot(gs[1, 0])
    ax_histx    = jfig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy    = jfig.add_subplot(gs[1, 1], sharey=ax)

    #Set custom axes options:
    ax_histx.tick_params(axis="both", which="both", bottom=False, top=False, left=True , right=False, labelbottom=False, labelleft=True )
    ax_histy.tick_params(axis="both", which="both", bottom=True , top=False, left=False, right=False, labelbottom=True , labelleft=False)

    for spine in ['top', 'right']:
        ax_histx.spines[spine].set_visible(False)
        ax_histy.spines[spine].set_visible(False)


    if is_neptune:  Target_Js = 1e6*np.array([3401.655e-6, -33.294e-6]); Sigma_Js = 1e6*np.array([   3.994e-6,  10.000e-6])
    else:           Target_Js = 1e6*np.array([3509.291e-6, -35.522e-6]); Sigma_Js = 1e6*np.array([   0.412e-6,   0.466e-6])
    
    ax.set_title(plottitlestr)
    ax.set_xlabel(r"$J_2  \ [10^{-6}]$")
    ax.set_ylabel(r"$J_4  \ [10^{-6}]$")

    J2 = 1e6*results['J2']
    J4 = 1e6*results['J4']

    x_array = J2
    y_array = J4

    x_array1 = J2
    y_array1 = J4

    if is_neptune:
        x_array2 = J2
        y_array2 = J4
        weightscorr = weights
    else:
        with open('result_dict_uranus_corr.pkl', 'rb') as f:
            resultscorr = pickle.load(f)   

        J2corr = 1e6*resultscorr['J2']
        J4corr = 1e6*resultscorr['J4']
        weightscorr = resultscorr["weights"]
        x_array2 = J2corr
        y_array2 = J4corr

    #2d histogram data:
    H1, xedges, yedges  = np.histogram2d(x_array1, y_array1, bins=bins,             weights=weights, density=True)
    H2, _, _            = np.histogram2d(x_array2, y_array2, bins=[xedges, yedges], weights=weightscorr, density=False)
    H1 = np.swapaxes(H1, 0, 1)
    H2 = np.swapaxes(H2, 0, 1)

    if False:
        #Hexagons:
        hb_template = ax.hexbin(
            x_array1, y_array1,
            gridsize=bins,
            extent=[-3, 3, -3, 3],
            C=None
        )

        hex_centers = hb_template.get_offsets()
        M = len(hex_centers)

        def aggregate_to_hexgrid(x, y, weights, centers):
            H = np.zeros(len(centers))
            idx = np.argmin((x[:,None]-centers[None,:,0])**2 +
                            (y[:,None]-centers[None,:,1])**2, axis=1)
            np.add.at(H, idx, weights)
            return H

        H1 = aggregate_to_hexgrid(x_array1, y_array1,
                                exp(x_array1, y_array1, cov=0.0000),
                                hex_centers)

        H2 = aggregate_to_hexgrid(x_array2, y_array2,
                                exp(x_array2, y_array2, cov=0.9861),
                                hex_centers)

    #Normalize:
    t1 = H1.copy()
    t2 = H2.copy()
    #t1 = np.log(t1+1)
    #t2 = np.log(t2+1)
    t1 /= np.max(t1)
    t2 /= np.max(t2)
    H1 = np.log(H1+1)
    H2 = np.log(H2+1)
    H1 /= np.max(H1)
    H2 /= np.max(H2)
    #H1 = np.stack([np.linspace(0,1,100)]*100, axis=0)
    #H2 = np.stack([np.linspace(0,1,100)]*100, axis=1)


    # Convert to RGBA using colormaps
    cmap1 = plt.get_cmap(colormaps[0])
    cmap2 = plt.get_cmap(colormaps[1])

    def blend_gamma(c1, c2, t1, t2, kappa):
        # mix them
        c1_lin = np.power(c1, kappa)
        c2_lin = np.power(c2, kappa)
        # use mixing weight (so that white gets mostly ignored)
        mixed_lin = (c1_lin * (t1[:,:,np.newaxis] + np.spacing(0)) + c2_lin * (t2[:,:,np.newaxis] + np.spacing(0)))/(t1[:,:,np.newaxis] + t2[:,:,np.newaxis] + 2*np.spacing(0))
        #magic = 1 + 4*t1*t2*(2 - (t1+t2))
        #mixed_lin /= magic[:,:,np.newaxis]
        mixed_lin = np.power(mixed_lin, 1/kappa)
        return mixed_lin

    #Mix the colors:
    rgba1 = cmap1(H1**gamma)
    rgba2 = cmap2(H2**gamma)
    #rgba1[:,:,3] = H1
    #rgba2[:,:,3] = H2
    mixed = blend_gamma(rgba1[:,:,:3], rgba2[:,:,:3], t1, t2, kappa)

    #Squares:

    # Display blended heatmap:
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    if is_neptune: ax.imshow(rgba1, extent=extent, origin='lower', aspect='auto')
    else:          ax.imshow(mixed, extent=extent, origin='lower', aspect='auto')

    if False:
        #Hexagons:

        # Display blended heatmap:
        hb_template.set_array(None)
        hb_template.set_facecolors(mixed)
        hb_template.set_edgecolors("none")

    #Plot the data:
    ax.errorbar(x=Target_Js[0], y=Target_Js[1], xerr=Sigma_Js[0], yerr=Sigma_Js[1], fmt='o', color='k', capsize=3)

    #Plot the x-data histogram:
    bins_x = np.linspace(x_array.min(), x_array.max(), bins_hist)
    ax_histx.hist(x_array, bins=bins_x, alpha=0.5, color=colors[0], weights=weights)
    counts, bin_edges = np.histogram(x_array, bins=bins_x, weights=weights)
    x_outline_x = np.repeat(bin_edges, 2)
    y_outline_x = np.hstack(([0], np.repeat(counts, 2), [0]))
    ax_histx.plot(x_outline_x, y_outline_x, color=colors[0])

    #Plot the y-data histogram:
    bins_y = np.linspace(y_array.min(), y_array.max(), bins_hist)
    ax_histy.hist(y_array, bins=bins_y, alpha=0.5, orientation='horizontal', color=colors[0], weights=weights)
    counts, bin_edges = np.histogram(y_array, bins=bins_y, weights=weights)
    y_outline_y = np.repeat(bin_edges, 2)
    x_outline_y = np.hstack(([0], np.repeat(counts, 2), [0]))
    ax_histy.plot(x_outline_y, y_outline_y, color=colors[0])


    if is_neptune == False:
        x_array = J2corr
        y_array = J4corr

        #Plot the x-data histogram:
        bins_x = np.linspace(x_array.min(), x_array.max(), bins_hist)
        ax_histx.hist(x_array, bins=bins_x, alpha=0.5, color=colors[1], weights=weightscorr)
        counts, bin_edges = np.histogram(x_array, bins=bins_x, weights=weightscorr)
        x_outline_x = np.repeat(bin_edges, 2)
        y_outline_x = np.hstack(([0], np.repeat(counts, 2), [0]))
        ax_histx.plot(x_outline_x, y_outline_x, color=colors[1])

        #Plot the y-data histogram:
        bins_y = np.linspace(y_array.min(), y_array.max(), bins_hist)
        ax_histy.hist(y_array, bins=bins_y, alpha=0.5, orientation='horizontal', color=colors[1], weights=weightscorr)
        counts, bin_edges = np.histogram(y_array, bins=bins_y, weights=weightscorr)
        y_outline_y = np.repeat(bin_edges, 2)
        x_outline_y = np.hstack(([0], np.repeat(counts, 2), [0]))
        ax_histy.plot(x_outline_y, y_outline_y, color=colors[1])

    jfig.savefig('plots/' + planetname + '/53J2J4correlation' + '_' + planetname + '.png', dpi=dpi)

    
    
    
    """
    # J2 vs J4 (relative error)

    saveaxs[5][4].set_title(plottitlestr)
    if is_neptune:  Target_Js = np.array([3401.655e-6, -33.294e-6]); Sigma_Js = np.array([   3.994e-6,  10.000e-6])
    else:           Target_Js = np.array([3509.291e-6, -35.522e-6]); Sigma_Js = np.array([   0.412e-6,   0.466e-6])
    saveaxs[5][4].scatter((results['J2'] - Target_Js[0])/Sigma_Js[0], (results['J4'] - Target_Js[1])/Sigma_Js[1], color = defaultcolor, s=0.5)
    saveaxs[5][4].errorbar(0, 0, 1, 1, marker = '*', color = 'k')
    saveaxs[5][4].set_xlabel(r"$J_2  \ [normalised]$")
    saveaxs[5][4].set_ylabel(r"$J_4  \ [normalised]$")
    savefig[5][4].savefig('plots/' + planetname + '/54J2J4correlation' + '_' + planetname + '.png', dpi=dpi)

    # J2 vs J4 (true)

    saveaxs[5][5].set_title(plottitlestr)
    saveaxs[5][5].scatter(1e6*results['J2'], 1e6*results['J4'], color = defaultcolor, s=0.5)
    if is_neptune:  Target_Js = 1e6*np.array([3401.655e-6, -33.294e-6]); Sigma_Js = 1e6*np.array([   3.994e-6,  10.000e-6])
    else:           Target_Js = 1e6*np.array([3509.291e-6, -35.522e-6]); Sigma_Js = 1e6*np.array([   0.412e-6,   0.466e-6])
    saveaxs[5][5].errorbar(x=Target_Js[0], y=Target_Js[1], xerr=Sigma_Js[0], yerr=Sigma_Js[1], marker = '*', color = 'k')
    saveaxs[5][5].set_xlabel(r"$J_2  \ [10^{-6}]$")
    saveaxs[5][5].set_ylabel(r"$J_4  \ [10^{-6}]$")
    savefig[5][5].savefig('plots/' + planetname + '/55J2J4correlation' + '_' + planetname + '.png', dpi=dpi)
    """
    #JUMPS

    #saveaxs[0][0].set_title('Average jump' + plottitlestr)
    saveaxs[0][0].set_title(plottitlestr)
    #saveaxs[0][0].bar(range(1023), results['raw jumps expl'], width=1, color = defaultcolor)
    saveaxs[0][0].plot(range(1023), results['raw jumps expl'], color = defaultcolor)
    saveaxs[0][0].invert_xaxis()
    saveaxs[0][0].set_xticks(xticklocations, xticklabels)
    saveaxs[0][0].set_xlim(left = 1023, right = 0)
    saveaxs[0][0].set_ylim(bottom = -0.2)
    saveaxs[0][0].set_xlabel("Normalised radius $r/R$")
    saveaxs[0][0].set_ylabel(r"$\overline{\Delta\rho}$ [kg m$^{-3}$]")

    savefig[0][0].savefig('plots/' + planetname + '/00rawjumps' + '_' + planetname + '.png', dpi=dpi)

    """
   # saveaxs[0,1].set_title('Raw Jumps of J explained (averaged)')
    saveaxs[0,1].plot(np.convolve(results['raw jumps expl'], np.ones(5)/5, mode = 'same'))
    """
    #saveaxs[0][1].set_title('Average jump intensity' + plottitlestr) #((real_jumplocation, jumpsize, jumpmagnitude))
    saveaxs[0][1].set_title(plottitlestr) #((real_jumplocation, jumpsize, jumpmagnitude))
    proc_jump_j_expl = results['processed jumps']
    jumpindex = proc_jump_j_expl[:, 0]
    jumpwidth = proc_jump_j_expl[:, 1]
    jumpheight = proc_jump_j_expl[:, 2]/results['nJ']
    jumpweight = jumpheight/jumpwidth
    resultweights = proc_jump_j_expl[:,3]
    resultweights /= np.mean(resultweights)
    #saveaxs[0][1].hist(jumpindex, bins = bincount, range = (0,N-1), weights = jumpweight, color = defaultcolor)
    y, x = np.histogram(jumpindex, bins = bincount, range = (0,N-1), weights = jumpweight*resultweights)
    saveaxs[0][1].plot(x[0:1023], y[0:1023], color = defaultcolor)
    saveaxs[0][1].invert_xaxis()
    saveaxs[0][1].set_xticks(xticklocations, xticklabels)
    saveaxs[0][1].set_xlim(left = 1023, right = 0)
    saveaxs[0][1].set_ylim(bottom = -0.05)
    saveaxs[0][1].set_xlabel("Normalised radius $r/R$")
    saveaxs[0][1].set_ylabel(r"$\overline{\Delta\rho / \Delta r}$  [$\Delta \ell^{-1}$ kg m$^{-3}$]")

    savefig[0][1].savefig('plots/' + planetname + '/01procjumps1' + '_' + planetname + '.png', dpi=dpi)

    #saveaxs[0][2].set_title('Jump widths, arbitrary units' + plottitlestr) #((real_jumplocation, jumpsize, jumpmagnitude))
    saveaxs[0][2].set_title(plottitlestr) #((real_jumplocation, jumpsize, jumpmagnitude))
    y, x = np.histogram(jumpindex, bins = bincount, range = (0,N-1), weights = jumpwidth/results['nJ']*resultweights)
    saveaxs[0][2].plot(x[0:1023], y[0:1023], color = defaultcolor)
    saveaxs[0][2].invert_xaxis()
    saveaxs[0][2].set_xticks(xticklocations, xticklabels)
    saveaxs[0][2].set_xlim(left = 1023, right = 0)
    saveaxs[0][2].set_ylim(bottom = -0.0005)
    saveaxs[0][2].set_xlabel("Normalised radius $r/R$")
    #saveaxs[0][2].get_yaxis().set_ticks([])
    saveaxs[0][2].set_ylabel(r"$\overline{\Delta r}$ [$\Delta \ell$]")


    savefig[0][2].savefig('plots/' + planetname + '/02procjumps2' + '_' + planetname + '.png', dpi=dpi)

    #saveaxs[0][3].set_title('Average jump magnitude' + plottitlestr) #((real_jumplocation, jumpsize, jumpmagnitude))
    saveaxs[0][3].set_title(plottitlestr) #((real_jumplocation, jumpsize, jumpmagnitude))
    y, x = np.histogram(jumpindex, bins = bincount, range = (0,N-1), weights = jumpheight*resultweights)
    saveaxs[0][3].plot(x[0:1023], y[0:1023], color = defaultcolor)
    saveaxs[0][3].invert_xaxis()
    saveaxs[0][3].set_xticks(xticklocations, xticklabels)
    saveaxs[0][3].set_xlim(left = 1023, right = 0)
    saveaxs[0][3].set_ylim(bottom = -0.2)
    saveaxs[0][3].set_xlabel("Normalised radius $r/R$")
    saveaxs[0][3].set_ylabel(r"$\overline{\Delta\rho}$ [kg m$^{-3}$]")

    savefig[0][3].savefig('plots/' + planetname + '/03procjumps3' + '_' + planetname + '.png', dpi=dpi)

    #saveaxs[1][2].set_title('Number of discontinuities in profile' + plottitlestr)
    saveaxs[1][2].set_title(plottitlestr)
    nr_jumps_J_expl = results['nr jumps']
    max_nr = np.max(nr_jumps_J_expl)
    nrjumpsbinvalues, nrjumpsbinedges, _ = saveaxs[1][2].hist(nr_jumps_J_expl, bins = max_nr + 1, range = (-0.5, max_nr+0.5), color = defaultcolor, density = True, weights=weights)
    #find index where we cumulate 95% of the total nr of jumps
    transform = trans.blended_transform_factory(saveaxs[1][2].transData, saveaxs[1][2].transAxes)
    saveaxs[1][2].vlines(np.percentile(nr_jumps_J_expl, 50, weights=weights, method='inverted_cdf'), ymin = 0, ymax = 1, linestyle ='-', color = 'black', transform = transform, label = 'Median')
    saveaxs[1][2].vlines(np.percentile(nr_jumps_J_expl, 95, weights=weights, method='inverted_cdf'), ymin = 0, ymax = 1, linestyle ='--', color = 'tab:gray', transform = transform, label = '95th percentile')
    saveaxs[1][2].vlines(np.mean(nr_jumps_J_expl*weights), ymin = 0, ymax = 1, linestyle ='-', color = 'tab:red', transform = transform, label = 'Average')
    saveaxs[1][2].xaxis.set_major_locator(MaxNLocator(integer=True))
    saveaxs[1][2].set_xlim(right=32)
    saveaxs[1][2].set_xlabel("Number of discontinuities")
    saveaxs[1][2].set_ylabel("Relative frequency")
    saveaxs[1][2].get_yaxis().set_ticks([])
    if is_neptune == False: saveaxs[1][2].legend()
    #savefig[1][2].subplots_adjust(left = 0.15)

    savefig[1][2].savefig('plots/' + planetname + '/12njumpexpl' + '_' + planetname + '.png', dpi=dpi)
    
    #saveaxs[1][3].set_title('Number of discontinuities in profile per criterion'+plottitlestr+'\nAverage and 95th percentile')
    saveaxs[1][3].set_title(plottitlestr)
    jumps_per_criteria = results['nr jumps per criterion']
    jumpcriteria # x           
    avg_jumps_per_criteria = [] # y1
    max_jumps_per_criteria = [] # y2
    for jumps in jumps_per_criteria:
        avg_jumps_per_criteria.append(np.mean(jumps*weights))
        max_jumps_per_criteria.append(np.percentile(jumps, 95, weights=weights, method='inverted_cdf'))        #find index which covers 95th percentile
    saveaxs[1][3].plot(avg_jumps_per_criteria, color = 'tab:red', label = 'Average')
    saveaxs[1][3].plot(max_jumps_per_criteria, linestyle ='--', color = 'tab:gray', label = '95th percentile')
    if is_neptune == False: saveaxs[1][3].legend()
    xticks = range(len(jumpcriteria))[::2]
    saveaxs[1][3].set_xticks(xticks)
    saveaxs[1][3].set_xticklabels(jumpcriteria[xticks])
    xticks = [0,1,3,5,7,9,11,13,15,17,19]
    xticks = [0,4,9,14,19]
    saveaxs[1][3].set_xticks(xticks)
    saveaxs[1][3].set_xticklabels(jumpcriteria[xticks])
    saveaxs[1][3].set_xlabel(r"Discontinuity criteria $c_D$ [$\Delta \ell^{-1}$ kg m$^{-3}$]")
    saveaxs[1][3].set_ylim(top=22)
    saveaxs[1][3].set_ylabel("Number of discontinuities")
    saveaxs[1][3].yaxis.set_major_locator(MaxNLocator(integer=True))

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
    """
    core_dens_v_max_jump_loc = results['core dens v max jump loc']
    core_dens_v_max_jump_loc = core_dens_v_max_jump_loc[core_dens_v_max_jump_loc[:,1]<450]
    core_dens_v_max_jump_loc = core_dens_v_max_jump_loc[core_dens_v_max_jump_loc[:,1]>280]
    x = 1-core_dens_v_max_jump_loc[:,1]/1024 #locations rescaled
    y = core_dens_v_max_jump_loc[:,0]
    saveaxs[0][4].hist2d(x,y, bins = 100, norm = 'log')
    linregress_result = linregress(x, y)
    
    print("Slope:")
    print(linregress_result.slope)
    print("R-value:")
    print(linregress_result.rvalue)
    print("P-value:")
    print(linregress_result.pvalue)
    print("Std-err:")
    print(linregress_result.stderr)
    
    xseq = np.linspace(0.55,0.75,num=100)
    #saveaxs[0][4].plot(xseq, a+b*xseq, color='k')
    #saveaxs[0][4].set_xticks(xticklocations, xticklabels)
    saveaxs[0][4].set_xlabel("Location of Discontinuity")
    saveaxs[0][4].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")

    savefig[0][4].savefig('plots/' + planetname + '/04coredensvmaxjumploc' + '_' + planetname + '.png', dpi=dpi)

    saveaxs[0][5].scatter(x,y, s=0.05)
    saveaxs[0][5].plot(xseq, linregress_result.intercept+linregress_result.slope*xseq, color='k')
    saveaxs[0][5].set_xlabel("Location of Discontinuity")
    saveaxs[0][5].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")
    savefig[0][5].savefig('plots/' + planetname + '/05coredensvmaxjumploc' + '_' + planetname + '.png', dpi=dpi)

    core_dens_v_jump_loc = results['core dens v jump loc']
    core_dens_v_jump_loc = core_dens_v_jump_loc[core_dens_v_jump_loc[:,1]<450]
    core_dens_v_jump_loc = core_dens_v_jump_loc[core_dens_v_jump_loc[:,1]>280]
    x = 1-core_dens_v_jump_loc[:,1]/1024 #locations rescaled
    y = core_dens_v_jump_loc[:,0]
    w = core_dens_v_jump_loc[:,2] #weights
    saveaxs[1][5].hist2d(x,y, bins = 100, weights = w, norm = 'log')
    saveaxs[1][5].set_xlabel("Location of Discontinuity")
    saveaxs[1][5].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")
    linregress_result = linregress(x, y)
    
    print("Slope:")
    print(linregress_result.slope)
    print("R-value:")
    print(linregress_result.rvalue)
    print("P-value:")
    print(linregress_result.pvalue)
    print("Std-err:")
    print(linregress_result.stderr)
    
    savefig[1][5].savefig('plots/' + planetname + '/15coredensvjumploc' + '_' + planetname + '.png', dpi=dpi)

    saveaxs[2][5].scatter(x,y, s=0.05)
    saveaxs[2][5].plot(xseq, linregress_result.intercept+linregress_result.slope*xseq, color='k')
    saveaxs[2][5].set_xlabel("Location of Discontinuity")
    saveaxs[2][5].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")
    savefig[2][5].savefig('plots/' + planetname + '/25coredensvjumploc' + '_' + planetname + '.png', dpi=dpi)



    #in mass
    core_dens_v_jump_loc = results['core dens v jump loc']
    core_dens_v_jump_loc = core_dens_v_jump_loc[core_dens_v_jump_loc[:,3]>0.4]
    x = core_dens_v_jump_loc[:,3]
    y = core_dens_v_jump_loc[:,0]
    saveaxs[3][5].hist2d(x,y, bins = 100, norm = 'log')
    saveaxs[3][5].set_xlabel("Enclosed Mass Fraction at First Discontinuity")
    saveaxs[3][5].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")
    linregress_result = linregress(x, y)
    
    print("Slope:")
    print(linregress_result.slope)
    print("R-value:")
    print(linregress_result.rvalue)
    print("P-value:")
    print(linregress_result.pvalue)
    print("Std-err:")
    print(linregress_result.stderr)
    print("Average & Median of Mass Location:")
    print(np.average(x))
    print(np.median(x))
    
    savefig[3][5].savefig('plots/' + planetname + '/35coredensvjumplocmass' + '_' + planetname + '.png', dpi=dpi)
    xseq = np.linspace(0.4,1,num=100)

    saveaxs[4][5].scatter(x,y, s=0.05)
    saveaxs[4][5].plot(xseq, linregress_result.intercept+linregress_result.slope*xseq, color='k')
    saveaxs[4][5].set_xlabel("Enclosed Mass Fraction at First Discontinuity")
    saveaxs[4][5].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")
    savefig[4][5].savefig('plots/' + planetname + '/45coredensvjumploc' + '_' + planetname + '.png', dpi=dpi)
    """
    #DISTR & CONTOUR

    #saveaxs[1][0].set_title('Distribution of density profiles, log scale' + plottitlestr)
    saveaxs[1][0].set_title(plottitlestr)
    log_grid = results['distr grid density']
    if is_neptune:  log_grid[log_grid < 1e-6] = 0
    else:           log_grid[log_grid < 3e-5] = 0
    logdistr = saveaxs[1][0].imshow((np.transpose(log_grid)), cmap = 'plasma', norm = col.LogNorm(), aspect = 'auto', origin = 'lower', extent = (-0.5, N-0.5, -0.5, rho_max-0.5))
    cbar = savefig[1][0].colorbar(logdistr, ax=saveaxs[1][0])
    cbar.set_label(label = 'Relative frequency', labelpad = 14)

    #literature values
    x = np.linspace(1,0,1024)
    if is_neptune:
        nettelmanNone = 1000*np.flip(np.loadtxt(r'planet_data/literature_values/Neptune/table_N1.txt')[:,4])
        nettelmanNonex = np.flip(np.loadtxt(r'planet_data/literature_values/Neptune/table_N1.txt')[:,2])
        nettelmanNonex /= nettelmanNonex[-1]
        nettelmanNtwo = 1000*np.flip(np.loadtxt(r'planet_data/literature_values/Neptune/table_N2b.txt')[:,4])
        nettelmanNtwox = np.flip(np.loadtxt(r'planet_data/literature_values/Neptune/table_N2b.txt')[:,2])
        nettelmanNtwox /= nettelmanNtwox[-1]
        saveaxs[1][0].plot(np.interp(x, nettelmanNonex, nettelmanNone), linestyle ='-', linewidth = 1, color = 'xkcd:marigold')
        saveaxs[1][0].plot(np.interp(x, nettelmanNtwox, nettelmanNtwo), linestyle ='--', linewidth = 1, color = 'xkcd:marigold')

    else: 
        nettelmanUone = 1000*np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/table_U1.txt')[:,4])
        nettelmanUonex = np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/table_U1.txt')[:,2])
        nettelmanUonex /= nettelmanUonex[-1]
        nettelmanUtwo = 1000*np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/table_U2.txt')[:,4])
        nettelmanUtwox = np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/table_U2.txt')[:,2])
        nettelmanUtwox /= nettelmanUtwox[-1]
        saveaxs[1][0].plot(np.interp(x, nettelmanUonex, nettelmanUone), linestyle ='-', linewidth = 1, color = 'xkcd:marigold')
        saveaxs[1][0].plot(np.interp(x, nettelmanUtwox, nettelmanUtwo), linestyle ='--', linewidth = 1, color = 'xkcd:marigold')
        helled = 1000*np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/density_ravit_1.txt')[:,1])
        helledx = np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/density_ravit_1.txt')[:,0])
        saveaxs[1][0].plot(np.interp(x, helledx, helled), linestyle ='-', linewidth = 1, color = 'xkcd:deep red')
        vazantwo = (np.loadtxt(r'planet_data/literature_values/Uranus/vazan_2.txt')[:,3])
        vazantwox = (np.loadtxt(r'planet_data/literature_values/Uranus/vazan_2.txt')[:,0])
        vazanthree = (np.loadtxt(r'planet_data/literature_values/Uranus/vazan_3.txt')[:,3])
        vazanthreex = (np.loadtxt(r'planet_data/literature_values/Uranus/vazan_3.txt')[:,0])
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
    #axins.imshow((np.transpose(log_grid)), cmap = 'plasma', norm = col.LogNorm(), aspect = 'auto', origin = 'lower', extent = (-0.5, N-0.5, -0.5, rho_max-0.5))
    axins.set_xticks([0,100,200,300], [1.0,0.9,0.8,0.7])
    axins.set_xlim(300,0)
    axins.set_ylim(0,2000)
    axins.set_yticks([0, 1000, 2000], [0, 1, 2])



    savefig[1][0].savefig('plots/' + planetname + '/10dens_distr' + '_' + planetname + '.png', dpi=2*dpi)


    #saveaxs[1][1].set_title('Distribution of density profiles, contour' + plottitlestr)
    saveaxs[1][1].set_title(plottitlestr)
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
                if cumulative_distr[x, y] > 0.025 and twolowersd[x] == -1: twolowersd[x] = y*rho_max/RES
                if cumulative_distr[x, y] > 0.16  and lowersd[x] == -1: lowersd[x] = y*rho_max/RES
                y += 1
            level += 1
        level -= 1
        while level > -1:
            while cumulative_distr[x, y] < 0.5+contourintervals[level]/2:
                percentile_grid[x, y] += level + 1
                if cumulative_distr[x, y] > 0.5   and contourmedian[x] == -1: contourmedian[x] = y*rho_max/RES
                if cumulative_distr[x, y] > 0.84  and uppersd[x] == -1: uppersd[x] = y*rho_max/RES
                y += 1
            level -= 1
        twouppersd[x] = y*rho_max/RES
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
    saveaxs[1][1].set_ylim(bottom=0)
    axins = inset_zoom_axes(saveaxs[1][1], [0.5, 0.5, 0.45, 0.45])
    axins.set_xticks([0,100,200,300], [1.0,0.9,0.8,0.7])
    axins.set_xlim(300,0)
    axins.set_ylim(0,2000)
    axins.set_yticks([0, 1000, 2000], [0, 1, 2])
    if is_neptune == False: saveaxs[1][1].legend(loc = 'upper left', fontsize = 14)

    savefig[1][1].savefig('plots/' + planetname + '/11contour' + '_' + planetname + '.png', dpi=dpi)



    #saveaxs[1][4].set_title('Distribution of pressure profiles, log scale' + plottitlestr)
    saveaxs[1][4].set_title(plottitlestr)
    log_grid = results['distr grid pressure']
    if is_neptune:  log_grid[log_grid < 1e-6] = 0
    else:           log_grid[log_grid < 3e-5] = 0
    logdistr = saveaxs[1][4].imshow((np.transpose(log_grid)), cmap = 'plasma', norm = col.LogNorm(), aspect = 'auto', origin = 'lower', extent = (-0.5, N-0.5, -0.5, p_max-0.5))
    cbar = savefig[1][4].colorbar(logdistr, ax=saveaxs[1][4])
    cbar.set_label(label = 'Relative frequency', labelpad = 14)

    #literature values
    x = np.linspace(1,0,1024)
    if is_neptune:
        nettelmanNone = 10**9*np.flip(np.loadtxt(r'planet_data/literature_values/Neptune/table_N1.txt')[:,1])
        nettelmanNonex = np.flip(np.loadtxt(r'planet_data/literature_values/Neptune/table_N1.txt')[:,2])
        nettelmanNonex /= nettelmanNonex[-1]
        nettelmanNtwo = 10**9*np.flip(np.loadtxt(r'planet_data/literature_values/Neptune/table_N2b.txt')[:,1])
        nettelmanNtwox = np.flip(np.loadtxt(r'planet_data/literature_values/Neptune/table_N2b.txt')[:,2])
        nettelmanNtwox /= nettelmanNtwox[-1]
        saveaxs[1][4].plot(np.interp(x, nettelmanNonex, nettelmanNone), linestyle ='-', linewidth = 1, color = 'xkcd:marigold')
        saveaxs[1][4].plot(np.interp(x, nettelmanNtwox, nettelmanNtwo), linestyle ='--', linewidth = 1, color = 'xkcd:marigold')

    else: 
        nettelmanUone = 10**9*np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/table_U1.txt')[:,1])
        nettelmanUonex = np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/table_U1.txt')[:,2])
        nettelmanUonex /= nettelmanUonex[-1]
        nettelmanUtwo = 10**9*np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/table_U2.txt')[:,1])
        nettelmanUtwox = np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/table_U2.txt')[:,2])
        nettelmanUtwox /= nettelmanUtwox[-1]
        saveaxs[1][4].plot(np.interp(x, nettelmanUonex, nettelmanUone), linestyle ='-', linewidth = 1, color = 'xkcd:marigold')
        saveaxs[1][4].plot(np.interp(x, nettelmanUtwox, nettelmanUtwo), linestyle ='--', linewidth = 1, color = 'xkcd:marigold')
        helled = np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/density_ravit_1.txt')[:,2])
        helledx = np.flip(np.loadtxt(r'planet_data/literature_values/Uranus/density_ravit_1.txt')[:,0])
        saveaxs[1][4].plot(np.interp(x, helledx, helled), linestyle ='-', linewidth = 1, color = 'xkcd:deep red')
        vazantwo = (np.loadtxt(r'planet_data/literature_values/Uranus/vazan_2.txt')[:,1])
        vazantwox = (np.loadtxt(r'planet_data/literature_values/Uranus/vazan_2.txt')[:,0])
        vazanthree = (np.loadtxt(r'planet_data/literature_values/Uranus/vazan_3.txt')[:,1])
        vazanthreex = (np.loadtxt(r'planet_data/literature_values/Uranus/vazan_3.txt')[:,0])
        saveaxs[1][4].plot(np.interp(x, vazantwox, vazantwo), linestyle ='-', linewidth = 1, color = 'xkcd:sky blue')
        saveaxs[1][4].plot(np.interp(x, vazanthreex, vazanthree), linestyle ='--', linewidth = 1, color = 'xkcd:sky blue')








    saveaxs[1][4].invert_xaxis()
    saveaxs[1][4].set_xticks(xticklocations, xticklabels)
    saveaxs[1][4].set_xlabel("Normalised radius $r/R$")
    saveaxs[1][4].set_ylim(bottom = -3e10, top = p_max)
    saveaxs[1][4].locator_params(axis='y', nbins=5)
    #saveaxs[1][4].set_yticks(yticklocations, yticklabels)
    saveaxs[1][4].set_ylabel(r"$p$ [Pa]")
    #axins = zoomed_inset_axes(saveaxs[1][4], zoom = 2, loc='upper right')
    axins = inset_zoom_axes(saveaxs[1][4], [0.5, 0.5, 0.4, 0.4])
    #axins.imshow((np.transpose(log_grid)), cmap = 'plasma', norm = col.LogNorm(), aspect = 'auto', origin = 'lower', extent = (-0.5, N-0.5, -0.5, rho_max-0.5))
    axins.set_xticks([0,100,200,300], [1.0,0.9,0.8,0.7])
    axins.set_xlim(300,0)
    axins.set_ylim(0,1e11)
    #axins.set_yticks([0, 1000, 2000], [0, 1, 2])

    savefig[1][4].savefig('plots/' + planetname + '/14press_distr' + '_' + planetname + '.png', dpi=2*dpi)

    """
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
        saveaxs[2][0].semilogy(np.linspace(0, rho_max, RES), averaged_sample, color = dev_colours[i])
        saveaxs[2][1].plot(np.linspace(0, rho_max, RES-1), averaged_first_derivative, color = dev_colours[i])
        saveaxs[2][2].plot(np.linspace(0, rho_max, RES-1), averaged_first_derivative_norm, color = dev_colours[i])
        #saveaxs[2,2].plot(np.linspace(0, rho_max, RES-2), averaged_second_derivative, color = dev_colours[i])
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
        saveaxs[4][0].semilogy(np.linspace(0, rho_max, RES), averaged_sample, color = dev_colours[i])
        saveaxs[4][1].plot(np.linspace(0, rho_max, RES-1), averaged_first_derivative, color = dev_colours[i])
        saveaxs[4][2].plot(np.linspace(0, rho_max, RES-1), averaged_first_derivative_norm, color = dev_colours[i])
        #saveaxs[2,2].plot(np.linspace(0, rho_max, RES-2), averaged_second_derivative, color = dev_colours[i])
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
    """




    #thiccness

    saveaxs[2][3].set_title('Size of confidence intervals' + plottitlestr)
    saveaxs[2][3].set_title(plottitlestr)
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
    saveaxs[2][3].set_ylim(bottom=0)
    saveaxs[2][3].set_xlim(left=1023, right=0)
    if is_neptune == False: saveaxs[2][3].legend()

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
    saveaxs[3][1].set_title(plottitlestr)
    maxdensbinvalues, maxdensbinedges, _  = saveaxs[3][1].hist(results['max dens'], bins = 400, color = defaultcolor, density = True, weights=weights)
    maxdensbinlocations = (maxdensbinedges[:-1]+maxdensbinedges[1:])/2
    
    modeindex = np.argmax(maxdensbinvalues)
    modelocation = maxdensbinlocations[modeindex]
    modevalue = maxdensbinvalues[modeindex]

    weighted_indices = maxdensbinvalues*np.arange(len(maxdensbinvalues))
    avgindex = int(np.sum(weighted_indices)/np.sum(maxdensbinvalues))
    avglocation = maxdensbinlocations[avgindex]
    avgvalue = maxdensbinvalues[avgindex]

    median = np.percentile(results['max dens'], 50, weights=weights, method='inverted_cdf')
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
    saveaxs[3][1].set_ylim(top = np.max(maxdensbinvalues)*1.1)
    saveaxs[3][1].set_xticks(yticklocations, yticklabels)
    saveaxs[3][1].set_xlabel(r"$\rho$ [1000 kg m$^{-3}$]")

    if is_neptune == False: saveaxs[3][1].legend()

    savefig[3][1].savefig('plots/' + planetname + '/31maxdens' + '_' + planetname + '.png', dpi=dpi)
    """
    #saveaxs[3][2].set_title('Average of profiles' + plottitlestr)
    saveaxs[3][2].set_title(plottitlestr)
    saveaxs[3][2].plot(results['avg start'], linestyle ='--', color = 'tab:gray', label = 'Generated')
    saveaxs[3][2].plot(results['avg respected start'], color = 'tab:purple', label = r'$\rho_{\text{max}}$ respected')
    saveaxs[3][2].plot(results['avg successful start'], color = 'tab:red', label = 'Successful start')
    saveaxs[3][2].plot(results['avg successful result'], color = 'tab:blue', label = 'Successful result')
    saveaxs[3][2].invert_xaxis()
    saveaxs[3][2].set_xticks(xticklocations, xticklabels)
    saveaxs[3][2].set_xlabel("Normalised radius $r/R$")
    saveaxs[3][2].set_yticks(yticklocations, yticklabels)
    saveaxs[3][2].set_ylabel(r"$\rho$ [1000 kg m$^{-3}$]")

    if is_neptune == False: saveaxs[3][2].legend()

    savefig[3][2].savefig('plots/' + planetname + '/32avgstartrespsuccres' + '_' + planetname + '.png', dpi=dpi)


    #saveaxs[3][3].set_title('Average change' + plottitlestr)
    saveaxs[3][3].set_title(plottitlestr)
    posneg = ['tab:blue' if y >= 0 else 'tab:red' for y in results['avg change']]
    saveaxs[3][3].bar(range(1024), results['avg change'], width=1, bottom = 0, color = posneg)
    saveaxs[3][3].axhline(0, linestyle ='--', color = 'tab:gray')
    saveaxs[3][3].invert_xaxis()
    saveaxs[3][3].set_xticks(xticklocations, xticklabels)
    saveaxs[3][3].set_xlabel("Normalised radius $r/R$")
    saveaxs[3][3].set_ylabel(r"$\Delta\rho$ [kg m$^{-3}$]")

    savefig[3][3].savefig('plots/' + planetname + '/33avgchange' + '_' + planetname + '.png', dpi=dpi)
    """

    #NMoI
    #saveaxs[4][3].set_title('Normalised Moments of Inertia' + plottitlestr)
    saveaxs[4][3].set_title(plottitlestr)
    maxnmoibinvalues, maxnmoibinedges, _  = saveaxs[4][3].hist(results['moments of inertia'], bins = 200, color = defaultcolor, density = True, weights=weights)
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
    if is_neptune: saveaxs[4][3].text(avglocation-0.0003, 1.05*avgvalue, '{:.5f}'.format(avglocation), ha='right', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small', backgroundcolor = 'none')
    else:          saveaxs[4][3].text(avglocation+0.00001, avgvalue/10, '{:.5f}'.format(avglocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small', backgroundcolor = 'none')

    #saveaxs[4][3].vlines(medianlocation, ymin = 0, ymax = medianvalue, linestyle ='-', color = 'black', label = 'Median')
    #saveaxs[4][3].text(medianlocation, medianvalue, '{:.2f}'.format(medianlocation/1000), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')

    saveaxs[4][3].set_ylabel('Relative frequency')
    #saveaxs[4][3].set_xticks(yticklocations, yticklabels)
    saveaxs[4][3].set_xlabel(r"Normalised Moment of Inertia $\lambda$ [unitless]")

    if is_neptune == False: saveaxs[4][3].legend()

    saveaxs[4][3].get_yaxis().set_ticks([])

    savefig[4][3].savefig('plots/' + planetname + '/43nmoi' + '_' + planetname + '.png', dpi=dpi)





    #FR
    #saveaxs[4][4].set_title('Flattening Ratios' + plottitlestr)
    saveaxs[4][4].set_title(plottitlestr)
    if is_neptune: offset = 1.00629; offsetstr = "1.00629"
    else: offset = 1.00680; offsetstr = "1.00680"
    maxfrbinvalues, maxfrbinedges, _  = saveaxs[4][4].hist((results['flattening ratios']-offset)*10**5, bins = 200, color = defaultcolor, density = True, weights=weights)
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
    if is_neptune: saveaxs[4][4].text(1.03*avglocation, avgvalue/10, '{:.3f}'.format(avglocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small', backgroundcolor = 'none')
    else:          saveaxs[4][4].text(1.004*avglocation, avgvalue/10, '{:.3f}'.format(avglocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small', backgroundcolor = 'none')

    """
    # reference & error bars
    fr_ref_neptune = (1.0058 - offset)*10**5
    fr_ref_neptune_err = 0.0010*10**5
    fr_ref_uranus = (1.0078 - offset)*10**5
    fr_ref_uranus_err = 0.0003*10**5
    if is_neptune:
        saveaxs[4][4].vlines(fr_ref_neptune, ymin = 0, ymax = avgvalue, linestyle ='-', color = 'k', label = 'Reference')
        saveaxs[4][4].errorbar(x=fr_ref_neptune, y=avgvalue*0.5, xerr=fr_ref_neptune_err, yerr=None, color = 'k', capsize=3, marker=None)
    else:         
        saveaxs[4][4].vlines(fr_ref_uranus, ymin = 0, ymax = avgvalue, linestyle ='-', color = 'k', label = 'Reference')
        saveaxs[4][4].errorbar(x=fr_ref_uranus, y=avgvalue*0.5, xerr=fr_ref_uranus_err, yerr=None, color = 'k', capsize=3, marker=None)

    # tof error
    err = avglocation*4*10**(-4)*10**5
    if is_neptune:
        saveaxs[4][4].errorbar(x=avglocation, y=avgvalue*0.6, xerr=err, yerr=None, color = 'tab:red', capsize=3, marker=None)
    else:         
        saveaxs[4][4].errorbar(x=avglocation, y=avgvalue*0.6, xerr=err, yerr=None, color = 'tab:red', capsize=3, marker=None)
    """
    #saveaxs[4][4].vlines(medianlocation, ymin = 0, ymax = medianvalue, linestyle ='-', color = 'black', label = 'Median')
    #saveaxs[4][4].text(medianlocation, medianvalue, '{:.2f}'.format(medianlocation), ha='left', va='bottom', rotation= 'horizontal', rotation_mode = 'anchor', fontsize = 'small')

    saveaxs[4][4].set_ylabel('Relative frequency')
    #saveaxs[4][4].set_xticks(yticklocations, yticklabels)
    saveaxs[4][4].ticklabel_format(useOffset=False)
    #TODO: BAD
    saveaxs[4][4].set_xlabel(f"$(r_f - {offsetstr}) \cdot 10^5$ [unitless]")

    if is_neptune == False: saveaxs[4][4].legend()

    saveaxs[4][4].get_yaxis().set_ticks([])

    savefig[4][4].savefig('plots/' + planetname + '/44fr' + '_' + planetname + '.png', dpi=dpi)




    """
    #3d plot
    x = np.arange(N)
    y = np.linspace(0, rho_max, RES)
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
