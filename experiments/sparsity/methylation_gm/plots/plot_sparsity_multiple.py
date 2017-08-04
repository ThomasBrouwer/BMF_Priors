"""
Plot the sparsity experiment outcomes.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
MSE_min, MSE_max = 1.5, 6.
fraction_min, fraction_max = 0.75, 1.0
fractions_unknown = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"sparsity_methylation_gm_multiple.png"


''' Load in the performances. '''
def eval_handle_nan(fin):
    string = open(fin,'r').readline()
    old, new = "nan", "10000" #"numpy.nan"
    string = string.replace(old, new)
    return eval(string)
    
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

nmf_np = eval_handle_nan(folder_results+'performances_baseline_mf_nonprobabilistic.txt')
row = eval(open(folder_results+'performances_baseline_average_row.txt','r').read())
column = eval(open(folder_results+'performances_baseline_average_column.txt','r').read())

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
    (nmf_np, 'Row', 'grey', '-', None),#'1'),
]
smallgraph1 = [
    (ggg,  'GGG',  'r', '-', '1'),
    (gggw, 'GGGW', 'r', '-', '3'),
    (ggga, 'GGGA', 'r', '-', '4'),
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
fig = plt.figure(figsize=(8,2.5))
fig.subplots_adjust(left=0.055, right=0.99, bottom=0.145, top=0.91)
ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((2, 4), (0, 2))
ax3 = plt.subplot2grid((2, 4), (0, 3))
ax4 = plt.subplot2grid((2, 4), (1, 2))
ax5 = plt.subplot2grid((2, 4), (1, 3))
axes = [ax1, ax2, ax3, ax4, ax5]

for ax in [ax1, ax2, ax3, ax4, ax5]:
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    ax.set_yticks(numpy.arange(0,MSE_max+1,1))
    ax.set_ylim(MSE_min, MSE_max)
    ax.set_xlim(fraction_min, fraction_max)
for ax in [ax2, ax3, ax4, ax5]:
    ax.set_xticks([]), ax.set_yticks([])
ax1.set_xlabel("K", fontsize=9)
ax1.set_ylabel("MSE", fontsize=9)
ax1.set_title("(a) All methods", fontsize=9)
ax2.set_title("(b) GGG, GGGA, GGGW", fontsize=9)
ax3.set_title("(c) GEE, GEEA, GL21", fontsize=9)
ax4.set_xlabel("(d) GLL, GLLI, GTT, \nGTTN, PGG, PGGG", fontsize=9, multialignment='center')
ax5.set_xlabel("(e) GGG, GVG, GEG, GVnG", fontsize=9)



''' Add the actual plots. '''
x = fractions_unknown
for i, performances_names_colours_linestyles_markers in enumerate([
    largegraph_performances_names_colours_linestyles_markers,
    smallgraph1, smallgraph2, smallgraph3, smallgraph4,
]):
    for performances, name, colour, linestyle, marker in performances_names_colours_linestyles_markers:
        y = numpy.mean(performances["MSE"],axis=1)
        axes[i].plot(x, y, label=name, linestyle=linestyle, marker=('$%s$' % marker if marker else ''), 
                     c=colour, markersize=5, linewidth=(2 if i==0 else 1))

plt.savefig(plot_file, dpi=600)