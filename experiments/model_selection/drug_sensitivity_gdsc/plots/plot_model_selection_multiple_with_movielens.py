"""
Plot the model selection experiment outcomes.
Have multiple plots:
- GDSC: One large with all methods
- GDSC: A few smaller ones comparing 2-4 methods
- ML100K: One larger with all methods
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
gdsc_MSE_min, gdsc_MSE_max = 660, 850
gdsc_values_K = [1,2,3,4,6,8,10,15,20,30]

ml100k_MSE_min, ml100k_MSE_max = 0.775, 1.2
ml100k_values_K = [1,2,3,4,6,8,10,15]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"model_selection_gdsc_multiple_with_movielens.png"


''' Set up the performances for the drug sensitivity data. '''
folder_results = "./../results/"
gdsc_ggg = eval(open(folder_results+'performances_gaussian_gaussian.txt','r').read())
gdsc_gggu = eval(open(folder_results+'performances_gaussian_gaussian_univariate.txt','r').read())
gdsc_gggw = eval(open(folder_results+'performances_gaussian_gaussian_wishart.txt','r').read())
gdsc_ggga = eval(open(folder_results+'performances_gaussian_gaussian_ard.txt','r').read())
gdsc_gll = eval(open(folder_results+'performances_gaussian_laplace.txt','r').read())
gdsc_glli = eval(open(folder_results+'performances_gaussian_laplace_ig.txt','r').read())
gdsc_gvg = eval(open(folder_results+'performances_gaussian_gaussian_volumeprior.txt','r').read())
gdsc_gvng = eval(open(folder_results+'performances_gaussian_gaussian_volumeprior_nonnegative.txt','r').read())
gdsc_geg = eval(open(folder_results+'performances_gaussian_gaussian_exponential.txt','r').read())
gdsc_gee = eval(open(folder_results+'performances_gaussian_exponential.txt','r').read())
gdsc_geea = eval(open(folder_results+'performances_gaussian_exponential_ard.txt','r').read())
gdsc_gtt = eval(open(folder_results+'performances_gaussian_truncatednormal.txt','r').read())
gdsc_gttn = eval(open(folder_results+'performances_gaussian_truncatednormal_hierarchical.txt','r').read())
gdsc_gl21 = eval(open(folder_results+'performances_gaussian_l21.txt','r').read())
gdsc_pgg = eval(open(folder_results+'performances_poisson_gamma.txt','r').read())
gdsc_pggg = eval(open(folder_results+'performances_poisson_gamma_gamma.txt','r').read())

gdsc_nmf_np = eval(open(folder_results+'performances_baseline_mf_nonprobabilistic.txt').read())

gdsc_largegraph_performances_names_colours_linestyles_markers = [
    (gdsc_ggg,  'GGG',  'r', '-', None),#'1'),
    (gdsc_gggu, 'GGGU', 'r', '-', None),#'2'),
    (gdsc_gggw, 'GGGW', 'r', '-', None),#'3'),
    (gdsc_ggga, 'GGGA', 'r', '-', None),#'4'),
    (gdsc_gll,  'GLL',  'r', '-', None),#'5'),
    (gdsc_glli, 'GLLI', 'r', '-', None),#'6'),
    (gdsc_gvg,  'GVG',  'r', '-', None),#'7'),
    (gdsc_gee,  'GEE',  'b', '-', None),#'1'),
    (gdsc_geea, 'GEEA', 'b', '-', None),#'2'),
    (gdsc_gtt,  'GTT',  'b', '-', None),#'3'),
    (gdsc_gttn, 'GTTN', 'b', '-', None),#'4'),
    (gdsc_gl21, 'GL21', 'b', '-', None),#'5'),
    (gdsc_geg,  'GEG',  'g', '-', None),#'1'),
    (gdsc_gvng, 'GVnG', 'g', '-', None),#'2'),
    (gdsc_pgg,  'PGG',  'y', '-', None),#'1'),
    (gdsc_pggg, 'PGGG', 'y', '-', None),#'2'),
    (gdsc_nmf_np, 'NMF-NP', 'grey', '-', None),#'1'),
]
gdsc_smallgraph1 = [
    (gdsc_ggg,  'GGG',  'r', '-', '1'),
    (gdsc_gggw, 'GGGW', 'r', '-', '3'),
    (gdsc_ggga, 'GGGA', 'r', '-', '4'),
]
gdsc_smallgraph2 = [
    (gdsc_gee,  'GEE',  'b', '-', '1'),
    (gdsc_geea, 'GEEA', 'b', '-', '2'),
    (gdsc_gl21, 'GL21', 'b', '-', '5'),
]
gdsc_smallgraph3 = [
    (gdsc_gll,  'GLL',  'r', '-', '5'),
    (gdsc_glli, 'GLLI', 'r', '-', '6'),
    (gdsc_gtt,  'GTT',  'b', '-', '3'),
    (gdsc_gttn, 'GTTN', 'b', '-', '4'),
    (gdsc_pgg,  'PGG',  'y', '-', '1'),
    (gdsc_pggg, 'PGGG', 'y', '-', '2'),
]
gdsc_smallgraph4 = [
    (gdsc_gvg,  'GVG',  'r', '-', '7'),
    (gdsc_gvng, 'GVnG', 'g', '-', '2'),
    (gdsc_ggg,  'GGG',  'r', '-', '1'),
    (gdsc_geg,  'GEG',  'g', '-', '1'),
]


''' Set up the performances for the drug sensitivity data. '''
folder_results = "./../../movielens_100K/results/"
ml100k_ggg = eval(open(folder_results+'performances_gaussian_gaussian.txt','r').read())
ml100k_gggu = eval(open(folder_results+'performances_gaussian_gaussian_univariate.txt','r').read())
ml100k_gggw = eval(open(folder_results+'performances_gaussian_gaussian_wishart.txt','r').read())
ml100k_ggga = eval(open(folder_results+'performances_gaussian_gaussian_ard.txt','r').read())
ml100k_gll = ml100k_ggg#eval(open(folder_results+'performances_gaussian_laplace.txt','r').read())
ml100k_glli = ml100k_gll #eval(open(folder_results+'performances_gaussian_laplace_ig.txt','r').read())
ml100k_gvg = eval(open(folder_results+'performances_gaussian_gaussian_volumeprior.txt','r').read())
ml100k_gvng = eval(open(folder_results+'performances_gaussian_gaussian_volumeprior_nonnegative.txt','r').read())
ml100k_geg = eval(open(folder_results+'performances_gaussian_gaussian_exponential.txt','r').read())
ml100k_gee = eval(open(folder_results+'performances_gaussian_exponential.txt','r').read())
ml100k_geea = eval(open(folder_results+'performances_gaussian_exponential_ard.txt','r').read())
ml100k_gtt = eval(open(folder_results+'performances_gaussian_truncatednormal.txt','r').read())
ml100k_gttn = eval(open(folder_results+'performances_gaussian_truncatednormal_hierarchical.txt','r').read())
ml100k_gl21 = eval(open(folder_results+'performances_gaussian_l21.txt','r').read())
ml100k_pgg = eval(open(folder_results+'performances_poisson_gamma.txt','r').read())
ml100k_pggg = eval(open(folder_results+'performances_poisson_gamma_gamma.txt','r').read())

ml100k_nmf_np = eval(open(folder_results+'performances_baseline_mf_nonprobabilistic.txt').read())

ml100k_largegraph_performances_names_colours_linestyles_markers = [
    (ml100k_ggg,  'GGG',  'r', '-', None),#'1'),
    (ml100k_gggu, 'GGGU', 'r', '-', None),#'2'),
    (ml100k_gggw, 'GGGW', 'r', '-', None),#'3'),
    (ml100k_ggga, 'GGGA', 'r', '-', None),#'4'),
    (ml100k_gll,  'GLL',  'r', '-', None),#'5'),
    (ml100k_glli, 'GLLI', 'r', '-', None),#'6'),
    (ml100k_gvg,  'GVG',  'r', '-', None),#'7'),
    (ml100k_gee,  'GEE',  'b', '-', None),#'1'),
    (ml100k_geea, 'GEEA', 'b', '-', None),#'2'),
    (ml100k_gtt,  'GTT',  'b', '-', None),#'3'),
    (ml100k_gttn, 'GTTN', 'b', '-', None),#'4'),
    (ml100k_gl21, 'GL21', 'b', '-', None),#'5'),
    (ml100k_geg,  'GEG',  'g', '-', None),#'1'),
    (ml100k_gvng, 'GVnG', 'g', '-', None),#'2'),
    (ml100k_pgg,  'PGG',  'y', '-', None),#'1'),
    (ml100k_pggg, 'PGGG', 'y', '-', None),#'2'),
    (ml100k_nmf_np, 'NMF-NP', 'grey', '-', None),#'1'),
]


''' Set up the plots. '''
fig = plt.figure(figsize=(10,2.5))
fig.subplots_adjust(left=0.045, right=0.995, bottom=0.145, top=0.915)
ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((2, 6), (0, 2))
ax3 = plt.subplot2grid((2, 6), (0, 3))
ax4 = plt.subplot2grid((2, 6), (1, 2))
ax5 = plt.subplot2grid((2, 6), (1, 3))
ax6 = plt.subplot2grid((2, 6), (0, 4), rowspan=2, colspan=2)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
axes_gdsc, axes_ml100k = [ax1, ax2, ax3, ax4, ax5], [ax6]

for ax in axes:
    (MSE_min, MSE_max) = (gdsc_MSE_min, gdsc_MSE_max) if ax in axes_gdsc else (ml100k_MSE_min, ml100k_MSE_max)
    values_K = gdsc_values_K if ax in axes_gdsc else ml100k_values_K
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(4)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    if ax in axes_ml100k:
        ax.set_yticks(numpy.arange(0,MSE_max+1,0.1))
    else:
        ax.set_yticks(range(0,MSE_max+1,50))
    ax.set_ylim(MSE_min,MSE_max)
    ax.set_xlim(0,values_K[-1]+1)
for ax in [ax2, ax3, ax4, ax5]:
    ax.set_xticks([]), ax.set_yticks([])
ax1.set_xlabel("K", fontsize=9)
ax1.set_ylabel("MSE", fontsize=9)
ax1.set_title("(a) All methods, GDSC drug sensitivity", fontsize=9)
ax2.set_title("(b) GGG, GGGW, GGGA", fontsize=9)
ax3.set_title("(c) GEE, GEEA, GL21", fontsize=9)
ax4.set_xlabel("(d) GLL, GLLI, GTT, \nGTTN, PGG, PGGG", fontsize=9, multialignment='center')
ax5.set_xlabel("(e) GGG, GVG, \nGEG, GVnG", fontsize=9, multialignment='center')
ax6.set_title("(f) All methods, MovieLens 100K", fontsize=9)



''' Add the actual plots. '''
for i, performances_names_colours_linestyles_markers in enumerate([
    gdsc_largegraph_performances_names_colours_linestyles_markers,
    gdsc_smallgraph1, gdsc_smallgraph2, gdsc_smallgraph3, gdsc_smallgraph4,
    ml100k_largegraph_performances_names_colours_linestyles_markers,
]):
    x = gdsc_values_K if i < 5 else ml100k_values_K
    for performances, name, colour, linestyle, marker in performances_names_colours_linestyles_markers:
        y = numpy.mean(performances["MSE"],axis=1)
        axes[i].plot(x, y, label=name, linestyle=linestyle, marker=('$%s$' % marker if marker else ''), 
                     c=colour, markersize=5, linewidth=(2 if i%5==0 else 1))

plt.savefig(plot_file, dpi=600)