'''
This file contains the updates for the Gibbs samplers, computing the conditional
posterior parameter values.

Updates for U, V - format (Likelihood) Prior:
- (Gaussian) Gaussian
- (Gaussian) Gaussian + Wishart
- (Gaussian) Gaussian + Automatic Relevance Determination
- (Gaussian) Gaussian + Volume Prior
- (Gaussian) Exponential
- (Gaussian) Exponential + ARD
- (Gaussian) Truncated Normal
- (Gaussian) Truncated Normal + hierarchical
- (Gaussian) Half Normal
- (Poisson)  Gamma
- (Poisson)  Gamma + hierarchical
- (Poisson)  Dirichlet

Other updates:
- mu, Sigma from Normal - Inverse Wishart [model: Gaussian + Wishart]
- lambdak from Automatic Relevance Determination [model: Gaussian + Automatic Relevance Determination]
- lambdak Automatic Relevance Determination [model: Exponential + Automatic Relevance Determination]
- mu, tau from hierarchical Truncated Normal [model: Truncated Normal + hierarchical]
- zij from Multinomial [model: all with Poisson likelihood]
- hi from hierarchical Gamma [model: Gamma + hierarchical]
'''


import numpy


''' (Gausian) Gaussian '''
def gaussian_gaussian_sigma():
    pass

def gaussian_gaussian_mu():
    pass

''' (Gausian) Gaussian + Wishart '''
def gaussian_gaussian_wishart_sigma():
    pass

def gaussian_gaussian_wishart_mu():
    pass

def gaussian_wishart_beta0():
    pass

def gaussian_wishart_v0():
    pass

def gaussian_wishart_mu0():
    pass

def gaussian_wishart_W0():
    pass

''' (Gausian) Gaussian + Automatic Relevance Determination '''
def gaussian_gaussian_ard_sigma():
    pass

def gaussian_gaussian_ard_mu():
    pass

def gaussian_ard_alpha():
    pass

def gaussian_ard_beta():
    pass

''' (Gausian) Gaussian + Volume Prior '''
def gaussian_gaussian_vp_sigma():
    pass

def gaussian_gaussian_vp_mu():
    pass

''' (Gausian) Exponential '''
def gaussian_exponential_tau():
    pass

def gaussian_exponential_mu():
    pass

''' (Gausian) Exponential + Automatic Relevance Determination '''
def gaussian_exponential_ard_tau():
    pass

def gaussian_exponential_ard_mu():
    pass

def exponential_ard_alpha():
    pass

def exponential_ard_beta():
    pass

''' (Gausian) Truncated Normal '''
def gaussian_tn_tau():
    pass

def gaussian_tn_mu():
    pass

''' (Gausian) Truncated Normal + hierarchical '''
def gaussian_tn_hierarchical_tau():
    pass

def gaussian_tn_hierarchical_mu():
    pass

def tn_hierarchical_mu_t():
    pass

def tn_hierarchical_mu_m():
    pass

def tn_hierarchical_tau_a():
    pass

def tn_hierarchical_tau_b():
    pass

''' (Gausian) Half Normal '''
def gaussian_hn_tau():
    pass

def gaussian_hn_mu():
    pass

''' (Poisson) Gamma '''
def poisson_gamma_alpha():
    pass

def poisson_gamma_beta():
    pass

def poisson_zij_n():
    pass

def poisson_zij_p():
    pass

''' (Poisson) Gamma + hierarchical '''
def poisson_gamma_hierarchical_alpha():
    pass

def poisson_gamma_hierarchical_beta():
    pass

def gamma_hierarchical_hi_alpha():
    pass

def gamma_hierarchical_hi_beta():
    pass

''' (Poisson) Dirichlet '''
def poisson_dirichlet_alpha():
    pass