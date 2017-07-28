"""
Make similarity kernels using the first repeat's factor matrices U, V.
Compute Gaussian similarity kernels (row-wise similarity) for U and V, as well
as using Spearman correlation.
"""

from helpers import construct_gaussian_kernel
from helpers import construct_correlation_kernel

import numpy


''' Load in the performances. '''
folder_results = "./../results/"
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
pgg_U = eval(open(folder_results+'poisson_gamma_U.txt','r').read())
pgg_V = eval(open(folder_results+'poisson_gamma_V.txt','r').read())
pggg_U = eval(open(folder_results+'poisson_gamma_gamma_U.txt','r').read())
pggg_V = eval(open(folder_results+'poisson_gamma_gamma_V.txt','r').read())
nmf_np_U = eval(open(folder_results+'baseline_mf_nonprobabilistic_U.txt','r').read())
nmf_np_V = eval(open(folder_results+'baseline_mf_nonprobabilistic_V.txt','r').read())


''' Do the analysis. '''
construct_kernel = construct_correlation_kernel #construct_gaussian_kernel
gaussian_name_kernelU_kernelU = [
    ('ggg',  construct_gaussian_kernel(ggg_U[0]),  construct_gaussian_kernel(ggg_V[0])),
    ('gggu', construct_gaussian_kernel(gggu_U[0]), construct_gaussian_kernel(gggu_V[0])),
    ('gggw', construct_gaussian_kernel(gggw_U[0]), construct_gaussian_kernel(gggw_V[0])),
    ('ggga', construct_gaussian_kernel(ggga_U[0]), construct_gaussian_kernel(ggga_V[0])),
    ('gvg',  construct_gaussian_kernel(gvg_U[0]),  construct_gaussian_kernel(gvg_V[0])),
    ('gee',  construct_gaussian_kernel(gee_U[0]),  construct_gaussian_kernel(gee_V[0])),
    ('geea', construct_gaussian_kernel(geea_U[0]), construct_gaussian_kernel(geea_V[0])),
    ('gtt',  construct_gaussian_kernel(gtt_U[0]),  construct_gaussian_kernel(gtt_V[0])),
    ('gttn', construct_gaussian_kernel(gttn_U[0]), construct_gaussian_kernel(gttn_V[0])),
    ('gvng', construct_gaussian_kernel(gvng_U[0]), construct_gaussian_kernel(gvng_V[0])),
    ('geg',  construct_gaussian_kernel(geg_U[0]),  construct_gaussian_kernel(geg_V[0])),
    ('pgg',  construct_gaussian_kernel(pgg_U[0]),  construct_gaussian_kernel(pgg_V[0])),
    ('pggg', construct_gaussian_kernel(pggg_U[0]), construct_gaussian_kernel(pggg_V[0])),
    ('nmf_np', construct_gaussian_kernel(nmf_np_U[0]), construct_gaussian_kernel(nmf_np_V[0])),
]
correlation_name_kernelU_kernelU = [
    ('ggg',  construct_correlation_kernel(ggg_U[0]),  construct_correlation_kernel(ggg_V[0])),
    ('gggu', construct_correlation_kernel(gggu_U[0]), construct_correlation_kernel(gggu_V[0])),
    ('gggw', construct_correlation_kernel(gggw_U[0]), construct_correlation_kernel(gggw_V[0])),
    ('ggga', construct_correlation_kernel(ggga_U[0]), construct_correlation_kernel(ggga_V[0])),
    ('gvg',  construct_correlation_kernel(gvg_U[0]),  construct_correlation_kernel(gvg_V[0])),
    ('gee',  construct_correlation_kernel(gee_U[0]),  construct_correlation_kernel(gee_V[0])),
    ('geea', construct_correlation_kernel(geea_U[0]), construct_correlation_kernel(geea_V[0])),
    ('gtt',  construct_correlation_kernel(gtt_U[0]),  construct_correlation_kernel(gtt_V[0])),
    ('gttn', construct_correlation_kernel(gttn_U[0]), construct_correlation_kernel(gttn_V[0])),
    ('gvng', construct_correlation_kernel(gvng_U[0]), construct_correlation_kernel(gvng_V[0])),
    ('geg',  construct_correlation_kernel(geg_U[0]),  construct_correlation_kernel(geg_V[0])),
    ('pgg',  construct_correlation_kernel(pgg_U[0]),  construct_correlation_kernel(pgg_V[0])),
    ('pggg', construct_correlation_kernel(pggg_U[0]), construct_correlation_kernel(pggg_V[0])),
    ('nmf_np', construct_correlation_kernel(nmf_np_U[0]), construct_correlation_kernel(nmf_np_V[0])),
]


''' Store the kernels. '''
folder_kernels = './kernels/'
for name, kernelU, kernelV in gaussian_name_kernelU_kernelU:
    numpy.savetxt(folder_kernels+'gaussian_%s_U.txt' % name, kernelU)
    numpy.savetxt(folder_kernels+'gaussian_%s_V.txt' % name, kernelV)
for name, kernelU, kernelV in correlation_name_kernelU_kernelU:
    numpy.savetxt(folder_kernels+'correlation_%s_U.txt' % name, kernelU)
    numpy.savetxt(folder_kernels+'correlation_%s_V.txt' % name, kernelV)