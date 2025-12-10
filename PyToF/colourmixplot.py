import numpy as np
import matplotlib.pyplot as plt

#Data for histograms:
bins = 120
bins_hist = 15
colors = ['C0', 'C1']
colormaps = ['Blues', 'Oranges']

#Fake Data:
N = 1024
data_linear_x = np.stack(N*[np.linspace(0,1,N)], axis=1)
data_linear_y = np.stack(N*[np.linspace(0,1,N)], axis=0)
H1 = data_linear_x
H2 = data_linear_y

#(Alternative) 2d histogram data:
#H1, xedges, yedges  = np.histogram2d(np.random.normal(0,1,10000), np.random.normal(0,1,10000), bins=bins,              density=True)
#H2, _, _            = np.histogram2d(np.random.normal(0.5,0.5,10000), np.random.normal(1,2,10000), bins=[xedges, yedges],  density=True)
#H1 = np.swapaxes(H1, 0, 1)
#H2 = np.swapaxes(H2, 0, 1)


#Create new figure:
fig = plt.figure(figsize=[6.4, 6.4], layout='constrained')
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.25), height_ratios=(1.25, 4), wspace=0.00, hspace=0.00)

#Define axes:
ax          = fig.add_subplot(gs[1, 0])
ax_histx    = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy    = fig.add_subplot(gs[1, 1], sharey=ax)

#Set custom axes options:
ax_histx.tick_params(axis="both", which="both", bottom=False, top=False, left=True , right=False, labelbottom=False, labelleft=True )
ax_histy.tick_params(axis="both", which="both", bottom=True , top=False, left=False, right=False, labelbottom=True , labelleft=False)

for spine in ['top', 'right']:
    ax_histx.spines[spine].set_visible(False)
    ax_histy.spines[spine].set_visible(False)

ax.set_title("Mixed Colours")
ax.set_xlabel("x increasing")
ax.set_ylabel("y increasing")

def blend_gamma(c1, c2, w1, w2, gamma):
    # expand weight for array broadcasting
    # need to concatenate the weights 4 times because of rgba
    w1 = np.stack(4*[w1], axis=-1)
    w2 = np.stack(4*[w2], axis=-1)
    # go from rgb (square root) colour space to linear (real) colour space
    c1_lin = np.power(c1, gamma)
    c2_lin = np.power(c2, gamma)
    # mix according to weights (so that white gets mostly ignored)
    # note this is slightly wrong as a full colour in x gets lightened through y as you go from full y to half y to no y
    # account for division by zero: where c1 has values, use them, otherwise use c2, if both are zero, use c2 (which is zero), if neither are zero, mix.
    mixed_lin = np.where(w1 != 0, c1_lin, c2_lin)
    mask = ((w1 != 0) & (w2 != 0))
    np.place(mixed_lin, mask, (c1_lin[mask] * w1[mask] + c2_lin[mask] * w2[mask]) / (w1[mask] + w2[mask]))

    # go back to rgb colour space
    mixed_lin = np.power(mixed_lin, 1/gamma)
    return mixed_lin

def mix_arrays(H1, H2, colormaps, gamma: float = 1, log: bool = False):
    #Create weights to weight the colour mixing by
    w1 = H1.copy()
    w2 = H2.copy()
    w1 /= np.max(w1)
    w2 /= np.max(w2)

    if log:
        #Values 0 need to stay at 0
        H1 = np.log(H1+1)
        H2 = np.log(H2+1)
    H1 /= np.max(H1)
    H2 /= np.max(H2)

    cmap1 = plt.get_cmap(colormaps[0])
    cmap2 = plt.get_cmap(colormaps[1])

    # Convert to RGBA using colormaps
    rgba1 = cmap1(H1**gamma)
    rgba2 = cmap2(H2**gamma)

    #Mix the colors:
    mixed = blend_gamma(rgba1, rgba2, w1, w2, 2)

    return mixed

mixed = mix_arrays(H1, H2, colormaps=colormaps, gamma=1, log=False)
# Display blended heatmap:
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
ax.imshow(mixed, origin='lower', aspect='auto')

fig.savefig('test')

