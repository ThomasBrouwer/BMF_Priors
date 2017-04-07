"""
Plot the sparsity experiment outcomes.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
MSE_min, MSE_max = 650, 1400
fractions_unknown = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"sparsity_gdsc.png"


''' Load in the performances. '''
ggg = eval(open(folder_results+'performances_gaussian_gaussian.txt','r').read())
gggu = eval(open(folder_results+'performances_gaussian_gaussian_univariate.txt','r').read())
gggw = eval(open(folder_results+'performances_gaussian_gaussian_wishart.txt','r').read())
ggga = eval(open(folder_results+'performances_gaussian_gaussian_ard.txt','r').read())
gvg = eval(open(folder_results+'performances_gaussian_gaussian_volumeprior.txt','r').read())
gvng = eval(open(folder_results+'performances_gaussian_gaussian_volumeprior_nonnegative.txt','r').read())
geg = eval(open(folder_results+'performances_gaussian_gaussian_exponential.txt','r').read())
gee = eval(open(folder_results+'performances_gaussian_exponential.txt','r').read())
geea = eval(open(folder_results+'performances_gaussian_exponential_ard.txt','r').read())
gtt = eval(open(folder_results+'performances_gaussian_truncatednormal.txt','r').read())
gttn = eval(open(folder_results+'performances_gaussian_truncatednormal_hierarchical.txt','r').read())
pgg = eval(open(folder_results+'performances_poisson_gamma.txt','r').read())
pggg = eval(open(folder_results+'performances_poisson_gamma_gamma.txt','r').read())

performances_names_colours_linestyles_markers = [
    (ggg,  'GGG',  'r', '-', 'o'),
    (gggu, 'GGGU', 'r', '-', 's'),
    (gggw, 'GGGW', 'r', '-', 'x'),
    (ggga, 'GGGA', 'r', '-', 'd'),
    (gvg,  'GVG',  'r', '-', '*'),
    (geg,  'GEG',  'g', '-', 'o'),
    (gvng, 'GVnG', 'g', '-', '*'),
    (gee,  'GEE',  'b', '-', 'o'),
    (geea, 'GEEA', 'b', '-', 'd'),
    (gtt,  'GTT',  'b', '-', 's'),
    (gttn, 'GTTN', 'b', '-', 'x'),
    (pgg,  'PGG',  'y', '-', 'o'),
    (pggg, 'PGGG', 'y', '-', 's'),
]


''' Plot the performances. '''
fig = plt.figure(figsize=(4,3))
fig.subplots_adjust(left=0.11, right=0.98, bottom=0.105, top=0.98)
plt.xlabel("Fraction missing", fontsize=12, labelpad=1)
plt.ylabel("MSE", fontsize=12, labelpad=1)

x = fractions_unknown
for performances, name, colour, linestyle, marker in performances_names_colours_linestyles_markers:
    y = numpy.mean(performances["MSE"],axis=1)
    plt.plot(x, y, label=name, linestyle=linestyle, marker=marker, c=colour, markersize=3)

plt.xticks(fontsize=6)
plt.yticks(numpy.arange(0,MSE_max+1,100),fontsize=6)
plt.ylim(MSE_min, MSE_max)
plt.xlim(fractions_unknown[0]-0.1, fractions_unknown[-1]+0.1)

plt.savefig(plot_file, dpi=600)