"""
Plot the sparsity experiment outcomes.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
gm_MSE_min, gm_MSE_max = 1.5, 6.
gm_fraction_min, gm_fraction_max = 0.75, 1.0
gm_fractions_unknown = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

gdsc_MSE_min, gdsc_MSE_max = 675, 1200
gdsc_fractions_unknown = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gdsc_fraction_min, gdsc_fraction_max = 0.38, 0.92

folder_plots = "./"
plot_file = folder_plots+"sparsity_methylation_gm_multiple_with_gdsc.png"


''' Load in the performances. '''
def eval_handle_nan(fin):
    string = open(fin,'r').readline()
    old, new = "nan", "10000" #"numpy.nan"
    string = string.replace(old, new)
    return eval(string)
    

''' Set up the performances for the methylation GM data. '''
folder_results = "./../results/"
gm_ggg = eval(open(folder_results+'performances_gaussian_gaussian.txt','r').read())
gm_gggu = eval(open(folder_results+'performances_gaussian_gaussian_univariate.txt','r').read())
gm_gggw = eval(open(folder_results+'performances_gaussian_gaussian_wishart.txt','r').read())
gm_ggga = eval(open(folder_results+'performances_gaussian_gaussian_ard.txt','r').read())
gm_gll = eval(open(folder_results+'performances_gaussian_laplace.txt','r').read())
gm_glli = eval(open(folder_results+'performances_gaussian_laplace_ig.txt','r').read())
gm_gvg = eval(open(folder_results+'performances_gaussian_gaussian_volumeprior.txt','r').read())
gm_gvng = eval(open(folder_results+'performances_gaussian_gaussian_volumeprior_nonnegative.txt','r').read())
gm_geg = eval(open(folder_results+'performances_gaussian_gaussian_exponential.txt','r').read())
gm_gee = eval(open(folder_results+'performances_gaussian_exponential.txt','r').read())
gm_geea = eval(open(folder_results+'performances_gaussian_exponential_ard.txt','r').read())
gm_gtt = eval(open(folder_results+'performances_gaussian_truncatednormal.txt','r').read())
gm_gttn = eval(open(folder_results+'performances_gaussian_truncatednormal_hierarchical.txt','r').read())
gm_gl21 = eval(open(folder_results+'performances_gaussian_l21.txt','r').read())
gm_pgg = eval(open(folder_results+'performances_poisson_gamma.txt','r').read())
gm_pggg = eval(open(folder_results+'performances_poisson_gamma_gamma.txt','r').read())

gm_nmf_np = eval_handle_nan(folder_results+'performances_baseline_mf_nonprobabilistic.txt')
gm_row = eval(open(folder_results+'performances_baseline_average_row.txt','r').read())
gm_column = eval(open(folder_results+'performances_baseline_average_column.txt','r').read())

gm_largegraph_performances_names_colours_linestyles_markers = [
    (gm_ggg,  'GGG',  'r', '-', None),#'1'),
    (gm_gggu, 'GGGU', 'r', '-', None),#'2'),
    (gm_gggw, 'GGGW', 'r', '-', None),#'3'),
    (gm_ggga, 'GGGA', 'r', '-', None),#'4'),
    (gm_gll,  'GLL',  'r', '-', None),#'5'),
    (gm_glli, 'GLLI', 'r', '-', None),#'6'),
    (gm_gvg,  'GVG',  'r', '-', None),#'7'),
    (gm_gee,  'GEE',  'b', '-', None),#'1'),
    (gm_geea, 'GEEA', 'b', '-', None),#'2'),
    (gm_gtt,  'GTT',  'b', '-', None),#'3'),
    (gm_gttn, 'GTTN', 'b', '-', None),#'4'),
    (gm_gl21, 'GL21', 'b', '-', None),#'5'),
    (gm_geg,  'GEG',  'g', '-', None),#'1'),
    (gm_gvng, 'GVnG', 'g', '-', None),#'2'),
    (gm_pgg,  'PGG',  'y', '-', None),#'1'),
    (gm_pggg, 'PGGG', 'y', '-', None),#'2'),
    (gm_nmf_np, 'Row', 'grey', '-', None),#'1'),
]
gm_smallgraph1 = [
    (gm_ggg,  'GGG',  'r', '-', '1'),
    (gm_gggu, 'GGGU', 'r', '-', '2'),
    (gm_gggw, 'GGGW', 'r', '-', '3'),
    (gm_ggga, 'GGGA', 'r', '-', '4'),
]
gm_smallgraph2 = [
    (gm_gee,  'GEE',  'b', '-', '1'),
    (gm_geea, 'GEEA', 'b', '-', '2'),
    (gm_geea, 'GL21', 'b', '-', '5'),
]


''' Set up the performances for the GDSC drug sensitivity data. '''
folder_results = "./../../drug_sensitivity_gdsc/results/"
gdsc_ggg = eval(open(folder_results+'performances_gaussian_gaussian.txt','r').read())
gdsc_gggu = eval(open(folder_results+'performances_gaussian_gaussian_univariate.txt','r').read())
gdsc_gggw = eval(open(folder_results+'performances_gaussian_gaussian_wishart.txt','r').read())
gdsc_ggga = eval(open(folder_results+'performances_gaussian_gaussian_ard.txt','r').read())
gdsc_gll = eval(open(folder_results+'performances_gaussian_laplace.txt','r').read())
gdsc_glli = gdsc_gll #eval(open(folder_results+'performances_gaussian_laplace_ig.txt','r').read())
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

gdsc_nmf_np = eval_handle_nan(folder_results+'performances_baseline_mf_nonprobabilistic.txt')
gdsc_row = eval(open(folder_results+'performances_baseline_average_row.txt','r').read())
gdsc_column = eval(open(folder_results+'performances_baseline_average_column.txt','r').read())

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
    (gdsc_nmf_np, 'Row', 'grey', '-', None),#'1'),
]
gdsc_smallgraph1 = [
    (gdsc_gvg,  'GVG',  'r', '-', '7'),
    (gdsc_gvng, 'GVnG', 'g', '-', '2'),
    (gdsc_ggg,  'GGG',  'r', '-', '1'),
    (gdsc_geg,  'GEG',  'g', '-', '1'),
]
gdsc_smallgraph2 = [
    (gdsc_gll,  'GLL',  'r', '-', '5'),
    (gdsc_glli, 'GLLI', 'r', '-', '6'),
    (gdsc_gtt,  'GTT',  'b', '-', '3'),
    (gdsc_gttn, 'GTTN', 'b', '-', '4'),
    (gdsc_pgg,  'PGG',  'y', '-', '1'),
    (gdsc_pggg, 'PGGG', 'y', '-', '2'),
]


''' Set up the plots. '''
fig = plt.figure(figsize=(10,2.5))
fig.subplots_adjust(left=0.035, right=0.98, bottom=0.145, top=0.915)
ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((2, 6), (0, 2))
ax3 = plt.subplot2grid((2, 6), (1, 2))
ax4 = plt.subplot2grid((2, 6), (0, 3), rowspan=2, colspan=2)
ax5 = plt.subplot2grid((2, 6), (0, 5))
ax6 = plt.subplot2grid((2, 6), (1, 5))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
axes_gm, axes_gdsc = [ax1, ax2, ax3], [ax4, ax5, ax6]

for ax in axes:
    (MSE_min, MSE_max) = (gm_MSE_min, gm_MSE_max) if ax in axes_gm else (gdsc_MSE_min, gdsc_MSE_max)
    (fraction_min, fraction_max) = (gm_fraction_min, gm_fraction_max) if ax in axes_gm else (gdsc_fraction_min, gdsc_fraction_max)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(4)
    if ax in axes_gm:
        ax.set_yticks(numpy.arange(0,MSE_max+1,1))
    else:
        ax.set_yticks(numpy.arange(0,MSE_max+1,100))
    ax.set_ylim(MSE_min, MSE_max)
    ax.set_xlim(fraction_min, fraction_max)
for ax in [ax2, ax3, ax5, ax6]:
    ax.set_xticks([]), ax.set_yticks([])
ax1.set_xlabel("Fraction unobserved", fontsize=9)
ax1.set_ylabel("MSE", fontsize=8)
ax4.set_xlabel("Fraction unobserved", fontsize=9)
ax1.set_title("(a) All methods, methylation GM", fontsize=9)
ax2.set_title("(b) GGG, GGGU, GGGA, GGGW", fontsize=9)
ax3.set_xlabel("(c) GEE, GEEA, GL21", fontsize=9)
ax4.set_title("(d) All methods, GDSC drug sensitivity", fontsize=9)
ax5.set_title("(e) GGG, GVG, GEG, GVnG", fontsize=9, multialignment='center')
ax6.set_xlabel("(f) GLL, GLLI, GTT, \nGTTN, PGG, PGGG", fontsize=9)


''' Add the actual plots. '''
for i, performances_names_colours_linestyles_markers in enumerate([
    gm_largegraph_performances_names_colours_linestyles_markers,
    gm_smallgraph1, gm_smallgraph2, 
    gdsc_largegraph_performances_names_colours_linestyles_markers,
    gdsc_smallgraph1, gdsc_smallgraph2, 
]):
    x = gm_fractions_unknown if i < 3 else gdsc_fractions_unknown
    for performances, name, colour, linestyle, marker in performances_names_colours_linestyles_markers:
        y = numpy.mean(performances["MSE"],axis=1)
        axes[i].plot(x, y, label=name, linestyle=linestyle, marker=('$%s$' % marker if marker else ''), 
                     c=colour, markersize=5, linewidth=(2 if (i%3)==0 else 1))

plt.savefig(plot_file, dpi=600)