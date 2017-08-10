"""
Plot the cross-validation experiment outcomes, in two columns.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import matplotlib.pyplot as plt
import numpy


''' Helpers for loading in the crossvalidation performances. '''
methods = ['GGG', 'GGGU', 'GGGW', 'GGGA', 'GLL', 'GLLI', 'GVG', 'GEE', 'GEEA',
           'GTT', 'GTTN', 'GL21', 'GEG', 'GVnG', 'PGG', 'PGGG', 'NMF-NP']
method_to_folder = {
    'GGG':  'gaussian_gaussian',
    'GGGU': 'gaussian_gaussian_univariate',
    'GGGW': 'gaussian_gaussian_wishart',
    'GGGA': 'gaussian_gaussian_ard',
    'GLL':  'gaussian_laplace',
    'GLLI': 'gaussian_laplace_ig',
    'GVG':  'gaussian_gaussian_volumeprior',
    'GEE':  'gaussian_exponential',
    'GEEA': 'gaussian_exponential_ard',
    'GTT':  'gaussian_truncatednormal',
    'GTTN': 'gaussian_truncatednormal_hierarchical',
    'GL21': 'gaussian_l21', 
    'GEG':  'gaussian_gaussian_exponential',
    'GVnG': 'gaussian_gaussian_volumeprior_nonnegative',
    'PGG':  'poisson_gamma',
    'PGGG': 'poisson_gamma_gamma',
    'NMF-NP': 'baseline_mf_nonprobabilistic',
}
datasets = ['GDSC IC50', 'CTRP EC50', 'CCLE IC50', 'CCLE EC50', 'MovieLens 100K', 
            'MovieLens 1M', 'Gene body methylation', 'Promoter methylation']
dataset_to_folder = {
    'GDSC IC50': 'drug_sensitivity_gdsc',
    'CTRP EC50': 'drug_sensitivity_ctrp', 
    'CCLE IC50': 'drug_sensitivity_ccle_ic',
    'CCLE EC50': 'drug_sensitivity_ccle_ec', 
    'MovieLens 100K': 'movielens_100K',
    'MovieLens 1M': 'movielens_1M', 
    'Gene body methylation': 'methylation_gm', 
    'Promoter methylation': 'methylation_pm',
}
dataset_to_MSEminmax = {
    'GDSC IC50': (0., 1.),
    'CTRP EC50': (0., 1.),
    'CCLE IC50': (0., 1.),
    'CCLE EC50': (0., 1.),
    'MovieLens 100K': (0., 1.),
    'MovieLens 1M': (0., 1.),
    'Gene body methylation': (0., 1.),
    'Promoter methylation': (0., 1.),
}
dataset_to_ylabel = {
    'GDSC IC50': 'GDSC IC50',
    'CTRP EC50': 'CTRP EC50',
    'CCLE IC50': 'CCLE IC50',
    'CCLE EC50': 'CCLE EC50',
    'MovieLens 1M': 'MovieLens \n1M',
    'MovieLens 100K': 'MovieLens \n100K',
    'Gene body methylation': 'Gene body \nmethylation',
    'Promoter methylation': 'Promoter \nmethylation',
}

def load_performances_from_file(loc):
    ''' Extract 5 performances from given file. '''
    text = open(loc, 'r').readlines()[1]
    performances = eval(text.split("'MSE': ")[1].split(", 'Rp': ")[0])
    return performances
    
def load_all_performances(folder_crossvalidation, dataset):
    ''' Load all methods' performances given the cross-validation folder of the dataset. '''
    all_performances = [
        load_performances_from_file(loc=(folder_crossvalidation+dataset_to_folder[dataset]+
                                         '/results/'+method_to_folder[method]+'/results.txt'))
        for method in methods
    ]
    return all_performances


''' Actually load in the performances. '''
folder_crossvalidation = project_location+'BMF_Priors/experiments/cross_validation/'
all_performances_datasets = [
    (dataset, load_all_performances(folder_crossvalidation, dataset=dataset))
    for dataset in [
        'GDSC IC50', 
        'CTRP EC50',
        'CCLE IC50', 
        'CCLE EC50', 
        'MovieLens 100K',
        #'MovieLens 1M',
        'Gene body methylation',
        'Promoter methylation',
    ]
]


''' Plot into one big graph. '''
fig = plt.figure(figsize=(8, 5))
fig.subplots_adjust(left=0.04, right=0.95, bottom=0.11, top=0.995)
ax1 = plt.subplot2grid((4, 2), (0, 0))
ax2 = plt.subplot2grid((4, 2), (1, 0))
ax3 = plt.subplot2grid((4, 2), (2, 0))
ax4 = plt.subplot2grid((4, 2), (3, 0))
ax5 = plt.subplot2grid((4, 2), (0, 1))
ax6 = plt.subplot2grid((4, 2), (1, 1))
ax7 = plt.subplot2grid((4, 2), (2, 1))
ax8 = plt.subplot2grid((4, 2), (3, 1))
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

# Turn off xticks and make ytick labels smaller
for ax in axes:
    ax.set_xticks([])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(7)

# Add method names to bottom plot
xlabels = methods
ax4.set_xticks(numpy.arange(0,len(xlabels),1))
ax4.set_xticklabels(xlabels, fontsize=8, rotation='vertical')
ax8.set_xticks(numpy.arange(0,len(xlabels),0.99) + 0.7)
ax8.set_xticklabels(xlabels, fontsize=8, rotation='vertical')

# Make the plots
for i, (dataset, all_performances) in enumerate(all_performances_datasets):
    x = numpy.arange(len(methods))
    y, err = numpy.mean(all_performances, axis=1), numpy.std(all_performances, axis=1)
    axes[i].errorbar(x=x, y=y, yerr=err, fmt='.', markersize=7, linewidth=1)
    axes[i].yaxis.set_label_position("right")
    axes[i].set_ylabel(dataset_to_ylabel[dataset], fontsize=10, rotation='vertical')

folder_plots = "./"
plot_file = folder_plots+"crossvalidation_twocolumns.png"
plt.savefig(plot_file, dpi=600)
