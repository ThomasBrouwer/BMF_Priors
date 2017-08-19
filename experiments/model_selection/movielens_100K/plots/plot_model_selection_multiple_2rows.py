"""
Plot the model selection experiment outcomes.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
MSE_min, MSE_max = 0.79, 1.125
values_K = [1,2,3,4,6,8,10,15]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"model_selection_movielens_100k_multiple_2rows.png"


''' Load in the performances. '''
ggg = eval(open(folder_results+'performances_gaussian_gaussian.txt','r').read())
gggu = eval(open(folder_results+'performances_gaussian_gaussian_univariate.txt','r').read())
gggw = eval(open(folder_results+'performances_gaussian_gaussian_wishart.txt','r').read())
ggga = eval(open(folder_results+'performances_gaussian_gaussian_ard.txt','r').read())
gll = eval(open(folder_results+'performances_gaussian_laplace.txt','r').read())
glli = eval(open(folder_results+'performances_gaussian_laplace_ig.txt','r').read())
gvg = eval(open(folder_results+'performances_gaussian_gaussian_volumeprior.txt','r').read())
gvng = eval(open(folder_results+'performances_gaussian_gaussian_volumeprior_nonnegative.txt','r').read())
geg = eval(open(folder_results+'performances_gaussian_gaussian_exponential.txt','r').read())
gee = eval(open(folder_results+'performances_gaussian_exponential.txt','r').read())
geea = eval(open(folder_results+'performances_gaussian_exponential_ard.txt','r').read())
gtt = eval(open(folder_results+'performances_gaussian_truncatednormal.txt','r').read())
gttn = eval(open(folder_results+'performances_gaussian_truncatednormal_hierarchical.txt','r').read())
gl21 = eval(open(folder_results+'performances_gaussian_l21.txt','r').read())
pgg = eval(open(folder_results+'performances_poisson_gamma.txt','r').read())
pggg = eval(open(folder_results+'performances_poisson_gamma_gamma.txt','r').read())

nmf_np = eval(open(folder_results+'performances_baseline_mf_nonprobabilistic.txt').read())

largegraph_performances_names_colours_linestyles_markers = [
    (ggg,  'GGG',  'r', '-', None),#'1'),
    (gggu, 'GGGU', 'r', '-', None),#'2'),
    (gggw, 'GGGW', 'r', '-', None),#'3'),
    (ggga, 'GGGA', 'r', '-', None),#'4'),
    (gll,  'GLL',  'r', '-', None),#'5'),
    (glli, 'GLLI', 'r', '-', None),#'6'),
    (gvg,  'GVG',  'r', '-', None),#'7'),
    (gee,  'GEE',  'b', '-', None),#'1'),
    (geea, 'GEEA', 'b', '-', None),#'2'),
    (gtt,  'GTT',  'b', '-', None),#'3'),
    (gttn, 'GTTN', 'b', '-', None),#'4'),
    (gl21, 'GL21', 'b', '-', None),#'5'),
    (geg,  'GEG',  'g', '-', None),#'1'),
    (gvng, 'GVnG', 'g', '-', None),#'2'),
    (pgg,  'PGG',  'y', '-', None),#'1'),
    (pggg, 'PGGG', 'y', '-', None),#'2'),
    (nmf_np, 'NMF-NP', 'grey', '-', None),#'1'),
]
smallgraph1 = [
    (ggg,  'GGG',  'r', '-', '1'),
    (gggw, 'GGGW', 'r', '-', '3'),
    (ggga, 'GGGA', 'r', '-', '4'),
    (gll,  'GLL',  'r', '-', '5'),
]
smallgraph2 = [
    (gee,  'GEE',  'b', '-', '1'),
    (geea, 'GEEA', 'b', '-', '2'),
    (gl21, 'GL21', 'b', '-', '5'),
]
smallgraph3 = [
    (gll,  'GLL',  'r', '-', '5'),
    (glli, 'GLLI', 'r', '-', '6'),
    (gtt,  'GTT',  'b', '-', '3'),
    (gttn, 'GTTN', 'b', '-', '4'),
    (pgg,  'PGG',  'y', '-', '1'),
    (pggg, 'PGGG', 'y', '-', '2'),
]
smallgraph4 = [
    (gvg,  'GVG',  'r', '-', '7'),
    (gvng, 'GVnG', 'g', '-', '2'),
    (ggg,  'GGG',  'r', '-', '1'),
    (geg,  'GEG',  'g', '-', '1'),
]


''' Set up the plots. '''
fig = plt.figure(figsize=(7,3))
ax1 = plt.subplot2grid((2, 3), (0, 0))
ax2 = plt.subplot2grid((2, 3), (0, 1), sharey=ax1)
ax3 = plt.subplot2grid((2, 3), (0, 2), sharey=ax1)
ax4 = plt.subplot2grid((2, 3), (1, 1), sharey=ax1)
ax5 = plt.subplot2grid((2, 3), (1, 2), sharey=ax1)
axes = [ax1, ax2, ax3, ax4, ax5]
fig.subplots_adjust(left=0.065, right=0.995, bottom=0.01, top=0.93, wspace=0.05)

# Set x and y limits, and label fontsizes
for ax in [ax1, ax2, ax3, ax4, ax5]:
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    ax.set_ylim(MSE_min,MSE_max)
    ax.set_xlim(0,values_K[-1]+1)

# Remove ticks and ticklabels of all plots except left one
for ax in [ax2, ax3, ax4, ax5]:
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

ax1.set_xlabel("K", fontsize=9, labelpad=-1)
ax1.set_ylabel("MSE", fontsize=9)
ax1.set_title("(k) All methods", fontsize=8)
ax2.set_title("(l) GGG, GGGW, GGGA, GLL", fontsize=8)
ax3.set_title("(m) GEE, GEEA, GL21", fontsize=8)
ax4.set_title("(n) GLL, GLLI, GTT, GTTN, PGG, PGGG", fontsize=8)
ax5.set_title("(o) GGG, GVG, GEG, GVnG", fontsize=8)


''' Add the actual plots. '''
x = values_K
for i, performances_names_colours_linestyles_markers in enumerate([
    largegraph_performances_names_colours_linestyles_markers,
    smallgraph1, smallgraph2, smallgraph3, smallgraph4,
]):
    for performances, name, colour, linestyle, marker in performances_names_colours_linestyles_markers:
        y = numpy.mean(performances["MSE"],axis=1)
        axes[i].plot(x, y, label=name, linestyle=linestyle, marker=('$%s$' % marker if marker else ''), 
                     c=colour, markersize=6, markevery=(5,2), linewidth=(2 if i==0 else 1))

plt.savefig(plot_file, dpi=600)