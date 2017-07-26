"""
Plot the convergence of the many different BMF algorithms on the MovieLens 100K data.
"""

import matplotlib.pyplot as plt


''' Plot settings. '''
MSE_min, MSE_max = 0.4, 1.2
iterations = range(1,200+1)

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"convergences_movielens100k.png"


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
ghh = eval(open(folder_results+'performances_gaussian_halfnormal.txt','r').read())
pgg = eval(open(folder_results+'performances_poisson_gamma.txt','r').read())
pggg = eval(open(folder_results+'performances_poisson_gamma_gamma.txt','r').read())

nmf_np = eval(open(folder_results+'performances_baseline_mf_nonprobabilistic.txt').read())
row = eval(open(folder_results+'performances_baseline_average_row.txt','r').read())
column = eval(open(folder_results+'performances_baseline_average_column.txt','r').read())


''' Assemble the average performances and method names. '''
performances_names_colours_linestyles_markers = [
    (ggg,  'GGG',  'r', '-', ''),
    (gggu, 'GGGU', 'r', '-', ''),
    (gggw, 'GGGW', 'r', '-', ''),
    (ggga, 'GGGA', 'r', '-', ''),
    (gvg,  'GVG',  'r', '-', ''),
    (geg,  'GEG',  'g', '-', ''),
    (gvng, 'GVnG', 'g', '-', ''),
    (gee,  'GEE',  'b', '-', ''),
    (geea, 'GEEA', 'b', '-', ''),
    (gtt,  'GTT',  'b', '-', ''),
    (gttn, 'GTTN', 'b', '-', ''),
#    (ghh,  'GHH',  'b', '-', ''),
    (pgg,  'PGG',  'y', '-', ''),
    (pggg, 'PGGG', 'y', '-', ''),
    (nmf_np, 'Row',    'grey', '-', ''),
    (column, 'NMF-NP', 'grey', '-', ''),
    (row,    'Col',    'grey', '-', ''),
]


''' Plot the performances. '''
fig = plt.figure(figsize=(3,2))
fig.subplots_adjust(left=0.105, right=0.975, bottom=0.125, top=0.975)
plt.xlabel("Iterations", fontsize=9, labelpad=0)
plt.ylabel("MSE", fontsize=9, labelpad=0)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)

x = iterations
for performances, name, colour, linestyle, marker in performances_names_colours_linestyles_markers:
    y = performances[:iterations[-1]]
    plt.plot(x, y, label=name, linestyle=linestyle, marker=marker, c=colour, 
             markersize=3, linewidth=1.2)
 
plt.ylim(MSE_min,MSE_max)
    
plt.savefig(plot_file, dpi=600)
    