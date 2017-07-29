"""
Plot the noise experiment outcomes, as a bar chart (grouped by NSR):
X-axis: noise level.
Y-axis: ratio of data variance to predictive error.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.experiments.noise.methylation_gm.data.create_load_noise_datasets import load_noise_datasets

import matplotlib.pyplot as plt
import numpy


''' Load the noisy datasets and noise levels. '''
noise_to_signal_ratios, Rs_noise, _ = load_noise_datasets()
variances = [R.var() for R in Rs_noise]


''' Plot settings. '''
y_min, y_max = 0., 17.
bar_width = 0.055
xtick_offset = 0.4 # to make xticks center of bars

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"noise_bar_methylation_gm.png"


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

nmf_np = eval(open(folder_results+'performances_baseline_mf_nonprobabilistic.txt','r').read())
row = eval(open(folder_results+'performances_baseline_average_row.txt','r').read())
column = eval(open(folder_results+'performances_baseline_average_column.txt','r').read())

performances_names_colours_linestyles_markers = [
    (ggg,  'GGG',  'r', '-', '.'),#'1'),
    (gggu, 'GGGU', 'r', '-', '.'),#'2'),
    (gggw, 'GGGW', 'r', '-', '.'),#'3'),
    (ggga, 'GGGA', 'r', '-', '.'),#'4'),
    (gvg,  'GVG',  'r', '-', '.'),#'5'),
    (geg,  'GEG',  'g', '-', '.'),#'1'),
    (gvng, 'GVnG', 'g', '-', '.'),#'2'),
    (gee,  'GEE',  'b', '-', '.'),#'1'),
    (geea, 'GEEA', 'b', '-', '.'),#'2'),
    (gtt,  'GTT',  'b', '-', '.'),#'3'),
    (gttn, 'GTTN', 'b', '-', '.'),#'4'),
    (pgg,  'PGG',  'y', '-', '.'),#'1'),
    (pggg, 'PGGG', 'y', '-', '.'),#'2'),
    (nmf_np, 'Row',    'grey', '-', '.'),#'1'),
    (column, 'NMF-NP', 'grey', '-', '.'),#'2'),
    (row,    'Col',    'grey', '-', '.'),#'3'),
]


''' Plot the performances. '''
fig = plt.figure(figsize=(5,2))
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.17, top=0.975)
plt.xlabel("Noise added (noise to signal ratio)", fontsize=9, labelpad=1)
plt.ylabel("Ratio data variance to error", fontsize=8, labelpad=1)

x = range(len(noise_to_signal_ratios))
for i, (performances, name, colour, linestyle, marker) in enumerate(performances_names_colours_linestyles_markers):
    ''' For each performance, compute the ratio of data variance to predictive error. '''
    y = numpy.mean(performances["MSE"],axis=1)
    y = [var/v for v, var in zip(y, variances)]
    x_i = [v+i*bar_width for v in x]
    plt.bar(x_i, y, bar_width, linestyle=linestyle, color=colour)

xlabels = ['%s%%' % int(NSR*100) for NSR in noise_to_signal_ratios]
plt.xticks(numpy.arange(len(noise_to_signal_ratios)) + xtick_offset, xlabels, fontsize=6)
plt.yticks(numpy.arange(0,y_max+1,1),fontsize=6)
plt.ylim(y_min, y_max)
plt.xlim(-0.2, len(noise_to_signal_ratios)+0.2)

plt.savefig(plot_file, dpi=600)