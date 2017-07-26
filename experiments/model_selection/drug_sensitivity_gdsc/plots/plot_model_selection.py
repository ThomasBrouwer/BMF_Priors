"""
Plot the model selection experiment outcomes
"""

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
MSE_min, MSE_max = 675, 850
values_K = [1,2,3,4,6,8,10,15,20,30]

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"model_selection_gdsc.png"
legend_file = folder_plots+'legend.png'


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
#    (ghh,  'GHH',  'b', '-', '*'),
    (pgg,  'PGG',  'y', '-', '1'),
    (pggg, 'PGGG', 'y', '-', '2'),
    (nmf_np, 'Row', 'grey', '-', '1'),
]


''' Plot the performances. '''
fig = plt.figure(figsize=(3,2))
fig.subplots_adjust(left=0.115, right=0.98, bottom=0.125, top=0.975)
plt.xlabel("K", fontsize=9, labelpad=1)
plt.ylabel("MSE", fontsize=9, labelpad=1)

x = values_K
for performances, name, colour, linestyle, marker in performances_names_colours_linestyles_markers:
    y = numpy.mean(performances["MSE"],axis=1)
    plt.plot(x, y, label=name, linestyle=linestyle, marker=('$%s$' % marker), c=colour, 
             markersize=5, linewidth=1.5)

plt.xticks(fontsize=6)
plt.yticks(range(0,MSE_max+1,50),fontsize=6)
plt.ylim(MSE_min,MSE_max)
plt.xlim(0,values_K[-1]+1)

plt.savefig(plot_file, dpi=600)


''' Set up the legend outside. '''
font_size_legend, number_of_columns, legend_box_line_width = 12, 7, 1
legend_line_width, legend_marker_size = 2, 2
ax = fig.add_subplot(111)
legend_fig = plt.figure(figsize=(8.8,0.63))
legend = legend_fig.legend(*ax.get_legend_handles_labels(), 
                           loc='center', ncol=number_of_columns,
                           prop={'size':font_size_legend}, markerscale=legend_marker_size,
                           fancybox=True, shadow=True)
legend.get_frame().set_linewidth(legend_box_line_width)
plt.setp(legend.get_lines(), linewidth=legend_line_width)
    
plt.savefig(legend_file, dpi=600)