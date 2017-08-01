"""
Same as make_kernels.py, but take the average of the kernel of each of the 10
runs.
"""

from helpers import construct_gaussian_kernel
from helpers import construct_Rp_correlation_kernel
from helpers import construct_Rs_correlation_kernel

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
gll_U = eval(open(folder_results+'gaussian_l21_U.txt','r').read())
gll_V = eval(open(folder_results+'gaussian_l21_V.txt','r').read())
pgg_U = eval(open(folder_results+'poisson_gamma_U.txt','r').read())
pgg_V = eval(open(folder_results+'poisson_gamma_V.txt','r').read())
pggg_U = eval(open(folder_results+'poisson_gamma_gamma_U.txt','r').read())
pggg_V = eval(open(folder_results+'poisson_gamma_gamma_V.txt','r').read())
nmf_np_U = eval(open(folder_results+'baseline_mf_nonprobabilistic_U.txt','r').read())
nmf_np_V = eval(open(folder_results+'baseline_mf_nonprobabilistic_V.txt','r').read())


''' Do the analysis. '''
construct_kernel = construct_Rs_correlation_kernel # construct_Rp_correlation_kernel # construct_gaussian_kernel # 
name_kernelU_kernelV = [
    ('ggg',    numpy.mean([construct_kernel(ggg_U[i])    for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(ggg_V[i])    for i in range(10)], axis=0)), 
    ('gggu',   numpy.mean([construct_kernel(gggu_U[i])   for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(gggu_V[i])   for i in range(10)], axis=0)), 
    ('gggw',   numpy.mean([construct_kernel(gggw_U[i])   for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(gggw_V[i])   for i in range(10)], axis=0)), 
    ('ggga',   numpy.mean([construct_kernel(ggga_U[i])   for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(ggga_V[i])   for i in range(10)], axis=0)), 
    ('gvg',    numpy.mean([construct_kernel(gvg_U[i])    for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(gvg_V[i])    for i in range(10)], axis=0)), 
    ('gee',    numpy.mean([construct_kernel(gee_U[i])    for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(gee_V[i])    for i in range(10)], axis=0)), 
    ('geea',   numpy.mean([construct_kernel(geea_U[i])   for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(geea_V[i])   for i in range(10)], axis=0)), 
    ('gtt',    numpy.mean([construct_kernel(gtt_U[i])    for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(gtt_V[i])    for i in range(10)], axis=0)), 
    ('gttn',   numpy.mean([construct_kernel(gttn_U[i])   for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(gttn_V[i])   for i in range(10)], axis=0)), 
    ('gvng',   numpy.mean([construct_kernel(gvng_U[i])   for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(gvng_V[i])   for i in range(10)], axis=0)), 
    ('geg',    numpy.mean([construct_kernel(geg_U[i])    for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(geg_V[i])    for i in range(10)], axis=0)), 
    ('gll',    numpy.mean([construct_kernel(gll_U[i])    for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(gll_V[i])    for i in range(10)], axis=0)), 
    ('pgg',    numpy.mean([construct_kernel(pgg_U[i])    for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(pgg_V[i])    for i in range(10)], axis=0)), 
    ('pggg',   numpy.mean([construct_kernel(pggg_U[i])   for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(pggg_V[i])   for i in range(10)], axis=0)), 
    ('nmf_np', numpy.mean([construct_kernel(nmf_np_U[i]) for i in range(10)], axis=0), 
               numpy.mean([construct_kernel(nmf_np_V[i]) for i in range(10)], axis=0)), 
]


''' Store the kernels. '''
folder_kernels = './average_kernels/'
kernel = ('gaussian' if construct_kernel == construct_gaussian_kernel else 'rp_correlation' 
          if construct_kernel == construct_Rp_correlation_kernel else 'rs_correlation')
for name, kernelU, kernelV in name_kernelU_kernelV:
    numpy.savetxt(folder_kernels+'%s_%s_U.txt' % (kernel, name), kernelU)
    numpy.savetxt(folder_kernels+'%s_%s_V.txt' % (kernel, name), kernelV)