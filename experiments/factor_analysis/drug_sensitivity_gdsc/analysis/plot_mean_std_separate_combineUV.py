"""
Plot the factor analysis outcomes:
- For each repeat, compute the mean and std of each factor k, for U and V separately
- Sort the factors by overall std (U[:,k]+V[:,k])
- Compute the average mean and average std, matching the factors by rank of overall std
- For each model compute the average mean and average std per factor (sorted), for U and V separately
- Plot these average means and averages per model, as a barchart
"""

from helpers import average_mean_std

import matplotlib.pyplot as plt
import numpy


''' Plot settings. '''
figsize = (8,5)
nrows, ncols = 8, 2
left, right, bottom, top = 0.04, 0.995, 0.01, 0.995
wspace, hspace = 0.125, None#0.1, 0.1
fontsize = 14

n_factors = 10
bar_width = 0.8
y_min, y_max = 0, 15.5
x_min, x_max = -0.25, n_factors+0.25

folder_plots = "./"
folder_results = "./../results/"
plot_file = folder_plots+"factor_analysis_separate_combineUV.png"


''' Load in the performances. '''
ggg_U = eval(open(folder_results+'gaussian_gaussian_U.txt','r').read())
ggg_V = eval(open(folder_results+'gaussian_gaussian_V.txt','r').read())
gggu_U = eval(open(folder_results+'gaussian_gaussian_univariate_U.txt','r').read())
gggu_V = eval(open(folder_results+'gaussian_gaussian_univariate_V.txt','r').read())
gggw_U = eval(open(folder_results+'gaussian_gaussian_wishart_U.txt','r').read())
gggw_V = eval(open(folder_results+'gaussian_gaussian_wishart_V.txt','r').read())
ggga_U = eval(open(folder_results+'gaussian_gaussian_ard_U.txt','r').read())
ggga_V = eval(open(folder_results+'gaussian_gaussian_ard_V.txt','r').read())
gvg_U = eval(open(folder_results+'gaussian_gaussian_volumeprior_U.txt','r').read())
gvg_V = eval(open(folder_results+'gaussian_gaussian_volumeprior_V.txt','r').read())
gvng_U = eval(open(folder_results+'gaussian_gaussian_volumeprior_nonnegative_U.txt','r').read())
gvng_V = eval(open(folder_results+'gaussian_gaussian_volumeprior_nonnegative_V.txt','r').read())
geg_U = eval(open(folder_results+'gaussian_gaussian_exponential_U.txt','r').read())
geg_V = eval(open(folder_results+'gaussian_gaussian_exponential_V.txt','r').read())
gee_U = eval(open(folder_results+'gaussian_exponential_U.txt','r').read())
gee_V = eval(open(folder_results+'gaussian_exponential_V.txt','r').read())
geea_U = eval(open(folder_results+'gaussian_exponential_ard_U.txt','r').read())
geea_V = eval(open(folder_results+'gaussian_exponential_ard_V.txt','r').read())
gtt_U = eval(open(folder_results+'gaussian_truncatednormal_U.txt','r').read())
gtt_V = eval(open(folder_results+'gaussian_truncatednormal_V.txt','r').read())
gttn_U = eval(open(folder_results+'gaussian_truncatednormal_hierarchical_U.txt','r').read())
gttn_V = eval(open(folder_results+'gaussian_truncatednormal_hierarchical_V.txt','r').read())
gll_U = eval(open(folder_results+'gaussian_l21_U.txt','r').read())
gll_V = eval(open(folder_results+'gaussian_l21_V.txt','r').read())
pgg_U = eval(open(folder_results+'poisson_gamma_U.txt','r').read())
pgg_V = eval(open(folder_results+'poisson_gamma_V.txt','r').read())
pggg_U = eval(open(folder_results+'poisson_gamma_gamma_U.txt','r').read())
pggg_V = eval(open(folder_results+'poisson_gamma_gamma_V.txt','r').read())
nmf_np_U = eval(open(folder_results+'baseline_mf_nonprobabilistic_U.txt','r').read())
nmf_np_V = eval(open(folder_results+'baseline_mf_nonprobabilistic_V.txt','r').read())


''' Do the analysis. '''
sort_by_std, use_absolute = True, True
name_meanU_stdU_meanV_stdV = [
    ('PGG',  average_mean_std(pgg_U,  pgg_V,  sort_by_std, use_absolute)),
    ('PGGG', average_mean_std(pggg_U, pggg_V, sort_by_std, use_absolute)),
    ('GGG',  average_mean_std(ggg_U,  ggg_V,  sort_by_std, use_absolute)),
    ('GGGU', average_mean_std(gggu_U, gggu_V, sort_by_std, use_absolute)),
    ('GGGW', average_mean_std(gggw_U, gggw_V, sort_by_std, use_absolute)),
    ('GGGA', average_mean_std(ggga_U, ggga_V, sort_by_std, use_absolute)),
    ('GVG',  average_mean_std(gvg_U,  gvg_V,  sort_by_std, use_absolute)),
    ('GVnG', average_mean_std(gvng_U, gvng_V, sort_by_std, use_absolute)),
    ('GEG',  average_mean_std(geg_U,  geg_V,  sort_by_std, use_absolute)),
    ('GEE',  average_mean_std(gee_U,  gee_V,  sort_by_std, use_absolute)),
    ('GEEA', average_mean_std(geea_U, geea_V, sort_by_std, use_absolute)),
    ('GTT',  average_mean_std(gtt_U,  gtt_V,  sort_by_std, use_absolute)),
    ('GTTN', average_mean_std(gttn_U, gttn_V, sort_by_std, use_absolute)),
    ('GLL',  average_mean_std(gll_U,  gll_V,  sort_by_std, use_absolute)),
    ('NMF-NP', average_mean_std(nmf_np_U, nmf_np_V, sort_by_std, use_absolute)),
]


''' Plot the performances. '''
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace, wspace=wspace)

x = numpy.arange(n_factors)
for i, (name, (avr_mean_U, avr_std_U, avr_mean_V, avr_std_V)) in enumerate(name_meanU_stdU_meanV_stdV):
    avr_mean = [(v1+v2)/2. for v1,v2 in zip(avr_mean_U,avr_mean_V)]    
    avr_std = [(v1+v2)/2. for v1,v2 in zip(avr_std_U,avr_std_V)]    
    
    axes[i/2][i%2].bar(x, avr_mean, bar_width, color='b')#, yerr=avr_std, error_kw=dict(ecolor='gray'))
    
    #average = ( sum(avr_mean_U) + sum(avr_mean_V) ) / float( len(avr_mean_U) + len(avr_mean_V) )
    #axes[i/2][i%2].axhline(y=average, c='grey')             
             
    axes[i/2][i%2].set_ylabel(name, fontsize=fontsize)
    axes[i/2][i%2].set_yticks([]), axes[i/2][i%2].set_xticks([])
    axes[i/2][i%2].set_ylim(y_min, y_max), axes[i/2][i%2].set_xlim(x_min, x_max)

# Turn off bottom right plot
axes[7][1].axis('off')

plt.savefig(plot_file, dpi=600)