"""
Plot the average Pearson correlation kernel of the absolute factor matrices,
across ten runs.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
sys.path.append(project_location)

from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50

import matplotlib.pyplot as plt
import numpy
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram


''' Plot settings. '''
figsize = (3.5,4)
left, right, bottom, top = 0.01, 0.99, 0.01, 0.99
fontsize = 14


''' Load in the kernels. '''
kernel_type = 'rp_correlation'
average_or_single = 'average' # 'single' # 
folder_kernels = './average_kernels_absolute/'

names_filenames = [
    ('GGG',  'ggg'),
    ('GGGU', 'gggu'),
    ('GGGW', 'gggw'),
    ('GGGA', 'ggga'),
    ('GLL',  'gll'),
    ('GLLI', 'glli'),
    ('GEG',  'geg'),
    ('GEE',  'gee'),
    ('GEEA', 'geea'),
    ('GTT',  'gtt'),
    ('GTTN', 'gttn'),
    ('GL21', 'gl21'),
    ('GVG',  'gvg'),
    ('GVnG', 'gvng'),
    ('PGG',  'pgg'),
    ('PGGG', 'pggg'),  
    ('NMF-NP', 'nmf_np'), 
]
name_plotname_kernelU_kernelV = [
    (name, filename, numpy.loadtxt(folder_kernels+'%s_%s_abs_U.txt' % (kernel_type, filename)), 
                     numpy.loadtxt(folder_kernels+'%s_%s_abs_V.txt' % (kernel_type, filename)))
    for name, filename in names_filenames
]


''' Print the averages of the kernels (non-diagonal values only). '''
print_averages = False
def average_kernel(K):
    return numpy.mean(numpy.extract(1.-numpy.eye(K.shape[0]), K))
    
if print_averages:
    for name, _, kernelU, kernelV in name_plotname_kernelU_kernelV:
        averageU, averageV = average_kernel(kernelU), average_kernel(kernelV)
        abs_averageU, abs_averageV = average_kernel(numpy.abs(kernelU)), average_kernel(numpy.abs(kernelV))
        print "Kerner average. Method: %s. \nU: %s. V: %s. U*V: %s. \nAbsU: %s. AbsV: %s. AbsU*AbsV: %s." % (
            name, averageU, averageV, averageU*averageV, 
            abs_averageU, abs_averageV, abs_averageU*abs_averageV)
                                 
        
''' Method for computing dendrogram. Return order of indices. '''
def compute_dendrogram(R):
    #plt.figure()
    # Hierarchical clustering methods: 
    # single (Nearest Point), complete (Von Hees), average (UPGMA), weighted (WPGMA), centroid (UPGMC), median (WPGMC), ward (incremental)
    Y = linkage(y=R, method='centroid', metric='euclidean') 
    Z = dendrogram(Z=Y, orientation='top', no_plot=True)#False)
    reordered_indices = Z['leaves']
    return reordered_indices
        

''' Plot the performances. '''
folder_plots_kernels = './plots_kernels_absolute/'
plot = True
# If True, run hierarchical clustering on R and reorder rows of kernelU, kernelV based on that
reorder_rows_columns = True 
if reorder_rows_columns:
    R, M = load_gdsc_ic50()
    indices_rows = compute_dendrogram(R)
    indices_columns = compute_dendrogram(R.T)

if plot:
    for name, plotname, kernelU, kernelV in name_plotname_kernelU_kernelV:
        # Reorder rows and columns using hierarchical clustering of R
        if reorder_rows_columns:
            kernelU = kernelU[indices_rows,:][:,indices_rows]
            kernelV = kernelV[indices_columns,:][:,indices_columns]
            kernelU, kernelV = numpy.abs(kernelU), numpy.abs(kernelV)
        
        ''' Plot U. '''
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(kernelU, cmap=plt.cm.bwr, interpolation='nearest', vmin=-1, vmax=1)
        
        # Axes labels
        #ax.set_xlabel("Drugs", fontsize=15)
        #ax.xaxis.set_label_position('bottom')
        
        # Turn off all the ticks
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        
        # Get rid of row and column labels
        ax.set_xticklabels([], minor=False, fontsize = 4)
        ax.set_yticklabels([], minor=False, fontsize = 4)
        
        # Store the plot
        plot_file = folder_plots_kernels+'%s_%s_%s_abs_U' % (average_or_single, kernel_type, plotname)
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        ''' Plot V. '''
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(kernelV, cmap=plt.cm.bwr, interpolation='nearest', vmin=-1, vmax=1)
        
        # Axes labels
        #ax.set_xlabel("Cell lines", fontsize=15)
        #ax.xaxis.set_label_position('bottom')
        
        # Turn off all the ticks
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        
        # Get rid of row and column labels
        ax.set_xticklabels([], minor=False, fontsize = 4)
        ax.set_yticklabels([], minor=False, fontsize = 4)
        
        # Store the plot
        plot_file = folder_plots_kernels+'%s_%s_%s_abs_V' % (average_or_single, kernel_type, plotname)
        plt.savefig(plot_file, dpi=600, bbox_inches='tight')
        plt.close()