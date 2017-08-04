"""
Plot the runtimes of the many different BMF algorithms on the GDSC drug sensitivity data.
"""

import matplotlib.pyplot as plt


''' Plot settings. '''
time_min, time_max = 0.0, 2.0
values_K = [5,10,20,50]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"runtimes_gdsc.png"


''' Load in the performances. '''
ggg = eval(open(folder_results+'times_gaussian_gaussian.txt','r').read())
gggu = eval(open(folder_results+'times_gaussian_gaussian_univariate.txt','r').read())
gggw = eval(open(folder_results+'times_gaussian_gaussian_wishart.txt','r').read())
ggga = eval(open(folder_results+'times_gaussian_gaussian_ard.txt','r').read())
gll = eval(open(folder_results+'times_gaussian_laplace.txt','r').read())
glli = eval(open(folder_results+'times_gaussian_laplace_ig.txt','r').read())
gvg = eval(open(folder_results+'times_gaussian_gaussian_volumeprior.txt','r').read())
gvng = eval(open(folder_results+'times_gaussian_gaussian_volumeprior_nonnegative.txt','r').read())
geg = eval(open(folder_results+'times_gaussian_gaussian_exponential.txt','r').read())
gee = eval(open(folder_results+'times_gaussian_exponential.txt','r').read())
geea = eval(open(folder_results+'times_gaussian_exponential_ard.txt','r').read())
gtt = eval(open(folder_results+'times_gaussian_truncatednormal.txt','r').read())
gttn = eval(open(folder_results+'times_gaussian_truncatednormal_hierarchical.txt','r').read())
gll = eval(open(folder_results+'times_gaussian_l21.txt','r').read())
#ghh = eval(open(folder_results+'times_gaussian_halfnormal.txt','r').read())
pgg = eval(open(folder_results+'times_poisson_gamma.txt','r').read())
pggg = eval(open(folder_results+'times_poisson_gamma_gamma.txt','r').read())

nmf_np = eval(open(folder_results+'times_baseline_mf_nonprobabilistic.txt').read())


''' Assemble the average performances and method names. '''
performances_names_colours_linestyles_markers = [
    (ggg,  'GGG',  'r', '-', '1'),
    (gggu, 'GGGU', 'r', '-', '2'),
    (gggw, 'GGGW', 'r', '-', '3'),
    (ggga, 'GGGA', 'r', '-', '4'),
    (gll,  'GLL',  'r', '-', '5'),
    (glli, 'GLLI', 'r', '-', '6'),
    (gvg,  'GVG',  'r', '-', '7'),
    (geg,  'GEG',  'g', '-', '1'),
    (gvng, 'GVnG', 'g', '-', '2'),
    (gee,  'GEE',  'b', '-', '1'),
    (geea, 'GEEA', 'b', '-', '2'),
    (gtt,  'GTT',  'b', '-', '3'),
    (gttn, 'GTTN', 'b', '-', '4'),
    (gll,  'GLL',  'b', '-', '5'),
#    (ghh,  'GHH',  'b', '-', '*'),
    (pgg,  'PGG',  'y', '-', '1'),
    (pggg, 'PGGG', 'y', '-', '2'),
    (nmf_np, 'Row', 'grey', '-', '1'),
]


''' Plot the performances. '''
fig = plt.figure(figsize=(3,2))
fig.subplots_adjust(left=0.11, right=0.98, bottom=0.125, top=0.98)
plt.xlabel("K", fontsize=9, labelpad=0)
plt.ylabel("Time (s)", fontsize=9, labelpad=0)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)

x = values_K
for performances, name, colour, linestyle, marker in performances_names_colours_linestyles_markers:
    y = performances
    plt.plot(x, y, label=name, linestyle=linestyle, marker=('$%s$' % marker), c=colour, 
             markersize=4, linewidth=1.5)
 
plt.ylim(time_min,time_max)
plt.xlim(values_K[0]-1,values_K[-1]+1)
    
plt.savefig(plot_file, dpi=600)
    