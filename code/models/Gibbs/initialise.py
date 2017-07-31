'''
This file contains methods for initialising random variables, such as U, V, tau, etc.
We either use the expectation or random draws of the prior distributions.
'''

from updates import update_tau_gaussian
from distributions.gamma import gamma_draw, gamma_mean
from distributions.normal import normal_draw, normal_mean
from distributions.multivariate_normal import multivariate_normal_draw, multivariate_normal_mean
from distributions.exponential import exponential_draw, exponential_mean
from distributions.truncated_normal import truncated_normal_draw, truncated_normal_mean
from distributions.half_normal import half_normal_draw, half_normal_mean
from distributions.normal_inverse_wishart import normal_inverse_wishart_draw, normal_inverse_wishart_mean
from distributions.multinomial import multinomial_draw, multinomial_mean
from distributions.dirichlet import dirichlet_draw, dirichlet_mean

import itertools
import numpy

def initialise_tau_gamma(alpha, beta, R, M, U, V):
    """ Initialise tau using the model updates. """
    return update_tau_gaussian(alpha=alpha, beta=beta, R=R, M=M, U=U, V=V)

def initialise_lamb_ard(init, K, alpha0, beta0):
    """ Initialise lamb (vector), with prior lamb_k ~ Gamma(alpha0,beta0). """
    initialise = gamma_draw if init == 'random' else gamma_mean
    lamb = numpy.zeros(K)
    for k in range(K):
        lamb[k] = initialise(alpha=alpha0, beta=beta0)
    return lamb

def initialise_U_gaussian(init, I, K, lamb):
    """ Initialise U, with prior Ui ~ N(0,I/lamb). """
    initialise = normal_draw if init == 'random' else normal_mean
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        lambdax = lamb[k] if isinstance(lamb, numpy.ndarray) else lamb
        U[i,k] = initialise(mu=0, tau=lambdax)
    return U

def initialise_U_gaussian_wishart(init, I, K, muU, sigmaU):
    """ Initialise U, with prior Ui ~ N(muU,sigmaU). """
    initialise = multivariate_normal_draw if init == 'random' else multivariate_normal_mean
    U = numpy.zeros((I,K))
    for i in range(I):
        U[i,:] = initialise(mu=muU, sigma=sigmaU)
    return U
    
def initialise_muU_sigmaU_wishart(init, mu0, beta0, v0, W0):
    """ Initialise muU and sigmaU (vectors), with prior muU, sigmaU ~ NIW(mu0,beta0,v0,W0). """
    K = mu0.shape[0]
    assert W0.shape == (K,K)
    initialise = normal_inverse_wishart_draw if init == 'random' else normal_inverse_wishart_mean
    muU, tauU = initialise(mu0=mu0, beta0=beta0, v0=v0, W0=W0)
    return (muU, tauU)

def initialise_U_exponential(init, I, K, lamb):
    """ Initialise U, with prior Uik ~ Exp(lamb). """
    initialise = exponential_draw if init == 'random' else exponential_mean
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        lambdax = lamb[k] if isinstance(lamb, numpy.ndarray) else lamb
        U[i,k] = initialise(lambdax=lambdax)
    return U

def initialise_U_truncatednormal(init, I, K, mu, tau):
    """ Initialise U, with prior Uik ~ TruncatedNormal(mu,tau). """
    initialise = truncated_normal_draw if init == 'random' else truncated_normal_mean
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        muUik = mu[i,k] if isinstance(mu, numpy.ndarray) else mu
        tauUik = tau[i,k] if isinstance(tau, numpy.ndarray) else tau
        U[i,k] = initialise(mu=muUik, tau=tauUik)
    return U
    
def initialise_muU_tauU_hierarchical(init, I, K, mu_mu, tau_mu, a, b):
    """ Initialise muU and tauU (matrices), with hierarchical prior proportional
        to N(muU_ik|mu_mu,tau_mu^-1) * Gamma(tauU_ik|a,b) * ...
        For simplicity we generate muU from N, and tauU from Gamma. """
    initialise_normal, initialise_gamma = ( (normal_draw, gamma_draw) if init == 'random'
                                            else (normal_mean, gamma_mean) )
    muU, tauU = numpy.zeros((I,K)), numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        muU[i,k] = initialise_normal(mu=mu_mu, tau=tau_mu)
        tauU[i,k] = initialise_gamma(alpha=a, beta=b)
    return (muU, tauU)

def initialise_U_halfnormal(init, I, K, sigma):
    """ Initialise U, with prior Uik ~ HalfNormal(sigma). """
    initialise = half_normal_draw if init == 'random' else half_normal_mean
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        U[i,k] = initialise(sigma=sigma)
    return U
    
def initialise_U_l21(init, I, K, lamb):
    """ Initialise U, with prior U ~ L21(lamb).
        We cannot sample from this prior, so we initialise U_ik ~ TN(0,lamb). """
    return initialise_U_truncatednormal(init=init, I=I, K=K, mu=0., tau=lamb)
    
def initialise_U_volumeprior(init, I, K, gamma):
    """ Initialise U, with prior U ~ VP(gamma).
        We cannot sample from this prior, so we initialise U_ik ~ N(0,1). """
    return initialise_U_gaussian(init=init, I=I, K=K, lamb=1.)
    
def initialise_U_volumeprior_nonnegative(init, I, K, gamma):
    """ Initialise U, with prior U ~ VP_nn(gamma).
        We cannot sample from this prior, so we initialise U_ik ~ TN(0,1). """
    return initialise_U_truncatednormal(init=init, I=I, K=K, mu=0., tau=1.)
    
def initialise_Z_multinomial(init, R, U, V):
    """ Initialise Z, with prior Zij ~ Multinomial(Rij, (Ui0*Vj0,..,UiK*VjK)). """
    I, J, K = R.shape[0], R.shape[1], U.shape[1]
    assert U.shape[0] == I and V.shape == (J,K)
    initialise = multinomial_draw if init == 'random' else multinomial_mean
    Z = numpy.zeros((I,J,K))
    for i,j in itertools.product(range(I),range(J)):
        p = U[i,:] * V[j,:]
        p /= p.sum()
        Z[i,j,:] = initialise(n=R[i,j], p=p)
    return Z

def initialise_U_gamma(init, I, K, a, b):
    initialise = gamma_draw if init == 'random' else gamma_mean
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        U[i,k] = initialise(alpha=a, beta=b)
    return U

def initialise_U_gamma_hierarchical(init, I, K, a, hU):
    initialise = gamma_draw if init == 'random' else gamma_mean
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        U[i,k] = initialise(alpha=a, beta=hU[i])
    return U
    
def initialise_hU_gamma_hierarchical(init, I, ap, bp):
    initialise = gamma_draw if init == 'random' else gamma_mean
    hU = numpy.zeros(I)
    for i in range(I):
        hU[i] = initialise(alpha=ap, beta=ap/bp)
    return hU

def initialise_U_dirichlet(init, I, K, alpha):
    initialise = dirichlet_draw if init == 'random' else dirichlet_mean
    assert alpha.shape == (K,)
    U = numpy.zeros((I,K))
    for i in range(I):
        U[i,:] = initialise(alpha=alpha)
    return U