'''
This file contains the updates for the variables, where we draw new values for
tau, U, V, etc.

Updates for U, V - format (Likelihood) Prior:
- (Gaussian) Gaussian (univariate posterior)
- (Gaussian) Gaussian (multivariate posterior)
- (Gaussian) Gaussian + Wishart
- (Gaussian) Gaussian + Automatic Relevance Determination
- (Gaussian) Volume Prior
- (Gaussian) Volume Prior (nonnegative)
- (Gaussian) Exponential
- (Gaussian) Exponential + ARD
- (Gaussian) Truncated Normal
- (Gaussian) Truncated Normal + hierarchical
- (Gaussian) Half Normal
- (Poisson)  Gamma
- (Poisson)  Gamma + hierarchical
- (Poisson)  Dirichlet

Other updates:
- tau (noise) from Gamma [model: all with Gaussian likelihood]
- mu, Sigma from Normal - Inverse Wishart [model: Gaussian + Wishart]
- lambdak from Automatic Relevance Determination [model: Gaussian + Automatic Relevance Determination]
- lambdak Automatic Relevance Determination [model: Exponential + Automatic Relevance Determination]
- mu, tau from hierarchical Truncated Normal [model: Truncated Normal + hierarchical]
- zij from Multinomial [model: all with Poisson likelihood]
- hi from hierarchical Gamma [model: Gamma + hierarchical]
'''

from parameters import gaussian_tau_alpha_beta
from parameters import gaussian_gaussian_mu_tau
from parameters import gaussian_gaussian_mu_sigma
from parameters import gaussian_gaussian_wishart_mu_sigma
from parameters import gaussian_wishart_beta0_v0_mu0_W0
from parameters import gaussian_gaussian_ard_mu_sigma
from parameters import gaussian_ard_alpha_beta
from parameters import gaussian_gaussian_volumeprior_mu_sigma
from parameters import gaussian_exponential_mu_tau
from parameters import gaussian_exponential_ard_mu_tau
from parameters import exponential_ard_alpha_beta
from parameters import gaussian_tn_mu_tau
from parameters import gaussian_tn_hierarchical_mu_tau
from parameters import tn_hierarchical_mu_m_t
from parameters import tn_hierarchical_tau_a_b
from parameters import gaussian_hn_mu_tau
from parameters import poisson_Z_n_p
from parameters import poisson_gamma_a_b
from parameters import poisson_gamma_hierarchical_a_b
from parameters import gamma_hierarchical_hUi_a_b
from parameters import poisson_dirichlet_alpha

from distributions.gamma import gamma_draw
from distributions.multivariate_normal import multivariate_normal_draw
from distributions.normal_inverse_wishart import normal_inverse_wishart_draw
from distributions.normal import normal_draw
from distributions.truncated_normal import truncated_normal_draw
from distributions.truncated_normal_vector import truncated_normal_vector_draw
from distributions.multinomial import multinomial_draw
from distributions.dirichlet import dirichlet_draw

import itertools
import minpy.numpy as numpy # import numpy


''' General Gaussian and Poisson models '''
def update_tau_gaussian(alpha, beta, R, M, U, V):
    """ Update tau (noise) in Gaussian models. """
    alpha_s, beta_s = gaussian_tau_alpha_beta(alpha, beta, R, M, U, V)
    new_tau = gamma_draw(alpha=alpha_s, beta=beta_s)
    return new_tau

#def update_Z_poisson(R, M, Omega, Z, U, V):
#    """ Update Z in Poisson models. """
#    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
#    n, p = poisson_Z_n_p(R=R, U=U, V=V)
#    for i,j in Omega:
#        Z[i,j,:] = multinomial_draw(n=n[i,j], p=p[i,j])
#    return Z
    
def update_Z_poisson(R, M, Omega, Z, U, V):
    """ Update Z in Poisson models. """
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    n_list, p_list = poisson_Z_n_p(R=R, U=U, V=V, Omega=Omega)
    for index,(i,j) in enumerate(Omega):
        Z[i,j,:] = multinomial_draw(n=n_list[index], p=p_list[index])
    return Z    
    
    
''' (Gausian) Gaussian (univariate posterior) '''
def update_U_gaussian_gaussian_univariate(lamb, R, M, U, V, tau):
    """ Update U for All Gaussian model (univariate posterior). """
    I, K = R.shape[0], V.shape[1]
    assert R.shape == M.shape and R.shape[1] == V.shape[0]
    for k in range(K):
        muUk, tauUk = gaussian_gaussian_mu_tau(k=k, lamb=lamb, R=R, M=M, U=U, V=V, tau=tau)
        for i in range(I):
            U[i,k] = normal_draw(mu=muUk[i], tau=tauUk[i])
    return U

def update_V_gaussian_gaussian_univariate(lamb, R, M, U, V, tau):  
    """ Update V for All Gaussian model (univariate posterior). """
    return update_U_gaussian_gaussian_univariate(lamb=lamb, R=R.T, M=M.T, U=V, V=U, tau=tau)


''' (Gausian) Gaussian (multivariate posterior) '''
def update_U_gaussian_gaussian_multivariate(lamb, R, M, V, tau):
    """ Update U for All Gaussian model (multivariate posterior). """
    I, K = R.shape[0], V.shape[1]
    assert R.shape == M.shape and R.shape[1] == V.shape[0]
    U = numpy.zeros((I,K))
    for i in range(I):
        muUi, sigmaUi = gaussian_gaussian_mu_sigma(lamb=lamb, Ri=R[i], Mi=M[i], V=V, tau=tau)
        U[i,:] = multivariate_normal_draw(mu=muUi, sigma=sigmaUi)
    return U
    
#def update_U_gaussian_gaussian_multivariate(lamb, R, M, V, tau):
#    """ Update U for All Gaussian model (multivariate posterior). """
#    I, K = R.shape[0], V.shape[1]
#    assert R.shape == M.shape and R.shape[1] == V.shape[0]
#    U = numpy.zeros((I,K))
#    muU, sigmaU = gaussian_gaussian_mu_sigma(lamb=lamb, R=R, M=M, V=V, tau=tau)
#    for i in range(I):
#        muUi, sigmaUi = gaussian_gaussian_mu_sigma_2(lamb, Ri=R[i], Mi=M[i], V=V, tau=tau)
#        assert numpy.array_equal(muU[i], muUi) and numpy.array_equal(sigmaU[i], sigmaUi)
#        U[i,:] = multivariate_normal_draw(mu=muU[i], sigma=sigmaU[i])
#    return U

def update_V_gaussian_gaussian_multivariate(lamb, R, M, U, tau):  
    """ Update V for All Gaussian model (multivariate posterior). """
    return update_U_gaussian_gaussian_multivariate(lamb=lamb, R=R.T, M=M.T, V=U, tau=tau)


''' (Gausian) Gaussian + Wishart '''
def update_U_gaussian_gaussian_wishart(muU, sigmaU, R, M, V, tau):
    """ Update U for All Gaussian + Wishart model. """
    I, K = R.shape[0], V.shape[1]
    assert R.shape == M.shape and R.shape[1] == V.shape[0]
    assert muU.shape == (K,) and sigmaU.shape == (K,K)
    sigmaU_inv = numpy.linalg.inv(sigmaU)
    U = numpy.zeros((I,K))
    for i in range(I):
        muUi, sigmaUi = gaussian_gaussian_wishart_mu_sigma(
            muU=muU, sigmaU_inv=sigmaU_inv, Ri=R[i], Mi=M[i], V=V, tau=tau)
        U[i,:] = multivariate_normal_draw(mu=muUi, sigma=sigmaUi)
    return U

def update_V_gaussian_gaussian_wishart(muV, sigmaV, R, M, U, tau):  
    """ Update V for All Gaussian + Wishart model. """
    return update_U_gaussian_gaussian_wishart(
        muU=muV, sigmaU=sigmaV, R=R.T, M=M.T, V=U, tau=tau)

def update_muU_sigmaU_gaussian_gaussian_wishart(mu0, beta0, v0, W0, U):
    """ Update muU and sigmaU for All Gaussian + Wishart model. """
    beta0_s, v0_s, mu0_s, W0_s = gaussian_wishart_beta0_v0_mu0_W0(
        beta0=beta0, v0=v0, mu0=mu0, W0=W0, U=U)
    new_muU, new_sigmaU = normal_inverse_wishart_draw(mu0=mu0_s,beta0=beta0_s,v0=v0_s,W0=W0_s)
    return (new_muU, new_sigmaU)

def update_muV_sigmaV_gaussian_gaussian_wishart(mu0, beta0, v0, W0, V):
    """ Update muV and sigmaV for All Gaussian + Wishart model. """
    return update_muU_sigmaU_gaussian_gaussian_wishart(
        mu0=mu0, beta0=beta0, v0=v0, W0=W0, U=V)
    

''' (Gausian) Gaussian + Automatic Relevance Determination '''
def update_U_gaussian_gaussian_multivariate_ard(lamb, R, M, V, tau):
    """ Update U for All Gaussian + ARD model. """
    I, K = R.shape[0], V.shape[1]
    assert R.shape == M.shape and R.shape[1] == V.shape[0]
    U = numpy.zeros((I,K))
    for i in range(I):
        muUi, sigmaUi = gaussian_gaussian_ard_mu_sigma(
            lamb=lamb, Ri=R[i], Mi=M[i], V=V, tau=tau)
        U[i,:] = multivariate_normal_draw(mu=muUi, sigma=sigmaUi)
    return U
    
def update_V_gaussian_gaussian_multivariate_ard(lamb, R, M, U, tau):
    """ Update V for All Gaussian + ARD model. """
    return update_U_gaussian_gaussian_multivariate_ard(lamb=lamb, R=R.T, M=M.T, V=U, tau=tau)

def update_lambda_gaussian_gaussian_ard(alpha0, beta0, U, V):
    """ Update lambda (vector) for All Gaussian + ARD model. """
    K = U.shape[1]
    new_lambda = numpy.zeros(K)
    for k in range(K):
        alpha_s, beta_s = gaussian_ard_alpha_beta(
            alpha0=alpha0, beta0=beta0, Uk=U[:,k], Vk=V[:,k])
        new_lambda[k] = gamma_draw(alpha=alpha_s, beta=beta_s)
    return new_lambda


''' (Gausian) Volume Prior '''
def update_U_gaussian_volumeprior(gamma, R, M, U, V, tau):
    """ Update U for Gaussian + Volume Prior model. """
    I, K = U.shape
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    for i,k in itertools.product(range(I),range(K)):
        muUik, tauUik = gaussian_gaussian_volumeprior_mu_sigma(
            i=i, k=k, gamma=gamma, Ri=R[i,:], Mi=M[i,:], U=U, V=V, tau=tau)
        U[i,k] = normal_draw(mu=muUik, tau=tauUik)
    return U
    
def update_V_gaussian_volumeprior(gamma, R, M, U, V, tau):
    """ Update V for Gaussian + Volume Prior model. """
    return update_U_gaussian_volumeprior(gamma=gamma, R=R.T, M=M.T, U=V, V=U, tau=tau)


''' (Gausian) Gaussian + Volume Prior '''
def update_U_gaussian_volumeprior_nonnegative(gamma, R, M, U, V, tau):
    """ Update U for Gaussian + nonnegative Volume Prior model. """
    I, K = U.shape
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    for i,k in itertools.product(range(I),range(K)):
        muUik, tauUik = gaussian_gaussian_volumeprior_mu_sigma(
            i=i, k=k, gamma=gamma, Ri=R[i,:], Mi=M[i,:], U=U, V=V, tau=tau)
        U[i,k] = truncated_normal_draw(mu=muUik, tau=tauUik)
    return U
    
def update_V_gaussian_volumeprior_nonnegative(gamma, R, M, U, V, tau):
    """ Update V for All Gaussian + nonnegative Volume Prior model. """
    return update_U_gaussian_volumeprior_nonnegative(
        gamma=gamma, R=R.T, M=M.T, U=V, V=U, tau=tau)


''' (Gausian) Exponential '''
def update_U_gaussian_exponential(lamb, R, M, U, V, tau):
    """ Update U for Gaussian + Exponential model. """
    I, K = U.shape
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    for k in range(K):
        muUk, tauUk = gaussian_exponential_mu_tau(k=k, lamb=lamb, R=R, M=M, U=U, V=V, tau=tau)
        U[:,k] = truncated_normal_vector_draw(mus=muUk, taus=tauUk)
    return U
    
def update_V_gaussian_exponential(lamb, R, M, U, V, tau):
    """ Update V for Gaussian + Exponential model. """
    return update_U_gaussian_exponential(lamb=lamb, R=R.T, M=M.T, U=V, V=U, tau=tau)
    

''' (Gausian) Exponential + Automatic Relevance Determination '''
def update_U_gaussian_exponential_ard(lamb, R, M, U, V, tau):
    """ Update U for Gaussian + Exponential + ARD model. """
    I, K = U.shape
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    for k in range(K):
        muUk, tauUk = gaussian_exponential_ard_mu_tau(k=k, lambdak=lamb[k], R=R, M=M, U=U, V=V, tau=tau)
        U[:,k] = truncated_normal_vector_draw(mus=muUk, taus=tauUk)
    return U
    
def update_V_gaussian_exponential_ard(lamb, R, M, U, V, tau):
    """ Update V for Gaussian + Exponential + ARD model. """
    return update_U_gaussian_exponential_ard(lamb=lamb, R=R.T, M=M.T, U=V, V=U, tau=tau)
    
def update_lambda_gaussian_exponential_ard(alpha0, beta0, U, V):
    """ Update lambda (vector) for Gaussian + Exponential + ARD model. """
    K = U.shape[1]
    new_lambda = numpy.zeros(K)
    for k in range(K):
        alpha_s, beta_s = exponential_ard_alpha_beta(
            alpha0=alpha0, beta0=beta0, Uk=U[:,k], Vk=V[:,k])
        new_lambda[k] = gamma_draw(alpha=alpha_s, beta=beta_s)
    return new_lambda
    

''' (Gausian) Truncated Normal '''
def update_U_gaussian_truncatednormal(muU, tauU, R, M, U, V, tau):
    """ Update U for Gaussian + Truncated Normal model. """
    I, K = U.shape
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    for k in range(K):
        muUk_s, tauUk_s = gaussian_tn_mu_tau(
            k=k, muU=muU, tauU=tauU, R=R, M=M, U=U, V=V, tau=tau)
        U[:,k] = truncated_normal_vector_draw(mus=muUk_s, taus=tauUk_s)
    return U
    
def update_V_gaussian_truncatednormal(muV, tauV, R, M, U, V, tau):
    """ Update V for Gaussian + Truncated Normal model. """
    return update_U_gaussian_truncatednormal(
        muU=muV, tauU=tauV, R=R.T, M=M.T, U=V, V=U, tau=tau)


''' (Gausian) Truncated Normal + hierarchical '''
def update_U_gaussian_truncatednormal_hierarchical(muU, tauU, R, M, U, V, tau):
    """ Update U for Gaussian + Truncated Normal + hierarchical model. """
    I, K = U.shape
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    for k in range(K):
        muUk_s, tauUk_s = gaussian_tn_hierarchical_mu_tau(
            k=k, muUk=muU[:,k], tauUk=tauU[:,k], R=R, M=M, U=U, V=V, tau=tau)
        U[:,k] = truncated_normal_vector_draw(mus=muUk_s, taus=tauUk_s)
    return U
    
def update_V_gaussian_truncatednormal_hierarchical(muV, tauV, R, M, U, V, tau):
    """ Update V for Gaussian + Truncated Normal + hierarchical model. """
    return update_U_gaussian_truncatednormal_hierarchical(
        muU=muV, tauU=tauV, R=R.T, M=M.T, U=V, V=U, tau=tau)
    
def update_muU_gaussian_truncatednormal_hierarchical(mu_mu, tau_mu, U, tauU):
    """ Update muU (matrix) for Gaussian + Truncated Normal + hierarchical model. """
    I, K = U.shape
    assert U.shape == tauU.shape
    (m_mu, t_mu) = tn_hierarchical_mu_m_t(mu_mu=mu_mu, tau_mu=tau_mu, U=U, tauU=tauU)
    new_muU = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        new_muU[i,k] = normal_draw(mu=m_mu[i,k], tau=t_mu[i,k])
    return new_muU

def update_muV_gaussian_truncatednormal_hierarchical(mu_mu, tau_mu, V, tauV):
    """ Update muV (matrix) for Gaussian + Truncated Normal + hierarchical model. """
    return update_muU_gaussian_truncatednormal_hierarchical(
        mu_mu=mu_mu, tau_mu=tau_mu, U=V, tauU=tauV)
    
def update_tauU_gaussian_truncatednormal_hierarchical(a, b, U, muU):
    """ Update tauU (matrix) for Gaussian + Truncated Normal + hierarchical model. """
    I, K = U.shape
    assert U.shape == muU.shape
    (a_s, b_s) = tn_hierarchical_tau_a_b(a=a, b=b, U=U, muU=muU)
    new_tauU = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        new_tauU[i,k] = gamma_draw(alpha=a_s[i,k], beta=b_s[i,k])
    return new_tauU

def update_tauV_gaussian_truncatednormal_hierarchical(a, b, V, muV):
    """ Update tauV (matrix) for Gaussian + Truncated Normal + hierarchical model. """
    return update_tauU_gaussian_truncatednormal_hierarchical(a=a, b=b, U=V, muU=muV)


''' (Gausian) Half Normal '''
def update_U_gaussian_halfnormal(sigma, R, M, U, V, tau):
    """ Update U for Gaussian + Half Normal model. """
    I, K = U.shape
    assert R.shape == M.shape and U.shape[0] == R.shape[0] and V.shape[0] == R.shape[1]
    for k in range(K):
        muUk_s, tauUk_s = gaussian_hn_mu_tau(
            k=k, sigma=sigma, R=R, M=M, U=U, V=V, tau=tau)
        U[:,k] = truncated_normal_vector_draw(mus=muUk_s, taus=tauUk_s)
    return U
    
def update_V_gaussian_halfnormal(sigma, R, M, U, V, tau):
    """ Update V for Gaussian + Half Normal model. """
    return update_U_gaussian_halfnormal(sigma=sigma, R=R.T, M=M.T, U=V, V=U, tau=tau)



''' (Poisson) Gamma '''
def update_U_poisson_gamma(a, b, M, V, Z):
    """ Update U for Poisson + Gamma model. """
    I, J, K = Z.shape
    assert V.shape == (J,K) and M.shape == (I,J)
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        (a_s, b_s) = poisson_gamma_a_b(a=a, b=b, Mi=M[i,:], Vk=V[:,k], Zik=Z[i,:,k]) 
        U[i,k] = gamma_draw(alpha=a_s, beta=b_s)
    return U
    
#def update_U_poisson_gamma(a, b, M, V, Z):
#    """ Update U for Poisson + Gamma model. """
#    I, J, K = Z.shape
#    assert V.shape == (J,K) and M.shape == (I,J)
#    U = numpy.zeros((I,K))
#    a_s, b_s = poisson_gamma_a_b(a=a, b=b, M=M, V=V, Z=Z) 
#    for i,k in itertools.product(range(I),range(K)):
#        U[i,k] = gamma_draw(alpha=a_s[i,k], beta=b_s[i,k])
#    return U
    
def update_V_poisson_gamma(a, b, M, U, Z):
    """ Update V for Poisson + Gamma model. """
    return update_U_poisson_gamma(a=a, b=b, M=M.T, V=U, Z=Z.transpose(1,0,2))
    
    
''' (Poisson) Gamma + hierarchical '''
def update_U_poisson_gamma_hierarchical(a, hU, M, V, Z):
    """ Update U for Poisson + Gamma + hierarchical model. """
    I, J, K = Z.shape
    assert hU.shape == (I,) and V.shape == (J,K)
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        (a_s, b_s) = poisson_gamma_hierarchical_a_b(
            a=a, hUi=hU[i], Mi=M[i,:], Vk=V[:,k], Zik=Z[i,:,k])
        U[i,k] = gamma_draw(alpha=a_s, beta=b_s)
    return U

def update_V_poisson_gamma_hierarchical(a, hV, M, U, Z):
    """ Update V for Poisson + Gamma + hierarchical model. """
    return update_U_poisson_gamma_hierarchical(a=a, hU=hV, M=M.T, V=U, Z=Z.transpose(1,0,2))
    
def update_hU_poisson_gamma_hierarchical(ap, bp, a, U):
    """ Update hU (vector) for Poisson + Gamma + hierarchical model. """
    I, K = U.shape
    hU = numpy.zeros(I)
    for i in range(I):
        (a_s, b_s) = gamma_hierarchical_hUi_a_b(ap=ap, bp=bp, a=a, Ui=U[i,:])
        hU[i] = gamma_draw(alpha=a_s, beta=b_s)
    return hU

def update_hV_poisson_gamma_hierarchical(ap, bp, a, V):
    """ Update hV (vector) for Poisson + Gamma + hierarchical model. """
    return update_hU_poisson_gamma_hierarchical(ap=ap, bp=bp, a=a, U=V)
    

''' (Poisson) Dirichlet '''
def update_U_poisson_dirichlet(alpha, M, Z):
    """ Update U for Poisson + Dirichlet model. """
    I, J, K = Z.shape
    assert M.shape == (I,J) and alpha.shape == (K,)
    U = numpy.zeros((I,K))
    for i in range(I):
        alpha_s = poisson_dirichlet_alpha(alpha=alpha, Mi=M[i], Zi=Z[i,:,:])
        U[i,:] = dirichlet_draw(alpha=alpha_s)
    return U
        
def update_V_poisson_dirichlet(alpha, M, Z):
    """ Update V for Poisson + Dirichlet model. """
    return update_U_poisson_dirichlet(alpha=alpha, M=M.T, Z=Z.transpose(1,0,2))