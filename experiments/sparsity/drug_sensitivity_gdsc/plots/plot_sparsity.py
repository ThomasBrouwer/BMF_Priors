"""
Plot the sparsity experiment outcomes.
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
MSE_min, MSE_max = 650, 1350
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
    (ggg,  'GGG',  'r', '-', '1'),
    (gggu, 'GGGU', 'r', '-', '2'),
    (gggw, 'GGGW', 'r', '-', '3'),
    (ggga, 'GGGA', 'r', '-', '4'),
    (gvg,  'GVG',  'r', '-', '5'),
    (geg,  'GEG',  'g', '-', '1'),
    (gvng, 'GVnG', 'g', '-', '2'),
    (gee,  'GEE',  'b', '-', '1'),
    (geea, 'GEEA', 'b', '-', '2'),
    (gtt,  'GTT',  'b', '-', '3'),
    (gttn, 'GTTN', 'b', '-', '4'),
    (pgg,  'PGG',  'y', '-', '1'),
    (pggg, 'PGGG', 'y', '-', '2'),
]


''' Plot the performances. '''
fig = plt.figure(figsize=(3,2))
fig.subplots_adjust(left=0.135, right=0.99, bottom=0.14, top=0.975)
plt.xlabel("Fraction missing", fontsize=9, labelpad=1)
plt.ylabel("MSE", fontsize=9, labelpad=1)

x = fractions_unknown
for performances, name, colour, linestyle, marker in performances_names_colours_linestyles_markers:
    y = numpy.mean(performances["MSE"],axis=1)
    plt.plot(x, y, label=name, linestyle=linestyle, marker=('$%s$' % marker), c=colour, 
             markersize=3, linewidth=0.8)

plt.xticks(fontsize=6)
plt.yticks(numpy.arange(0,MSE_max+1,100),fontsize=6)
plt.ylim(MSE_min, MSE_max)
plt.xlim(fractions_unknown[0]-0.05, fractions_unknown[-1]+0.05)

plt.savefig(plot_file, dpi=600)