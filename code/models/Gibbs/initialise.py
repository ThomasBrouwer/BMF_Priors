'''
This file contains methods for initialising random variables, such as U, V, tau, etc.
We either use the expectation or random draws of the prior distributions.
'''

from updates import update_tau_gaussian
from distributions.gamma import gamma_draw, gamma_mean
from distributions.normal import normal_draw, normal_mean
from distributions.exponential import exponential_draw, exponential_mean
from distributions.truncated_normal import truncated_normal_draw, truncated_normal_mean
from distributions.half_normal import half_normal_draw, half_normal_mean

import itertools
import numpy


def initialise_tau_gamma(alpha, beta, R, M, U, V):
    """ Initialise tau using the model updates. """
    return update_tau_gaussian(alpha=alpha, beta=beta, R=R, M=M, U=U, V=V)

def initialise_Z_multinomial():
    #TODO:
    pass

def initialise_lamb_ard(init, K, alpha0, beta0):
    """ Initialise lamb (vector), with prior lamb_k ~ Gamma(alpha0,beta0). """
    lamb = numpy.zeros(K)
    for k in range(K):
        lamb[k] = ( gamma_draw(alpha=alpha0, beta=beta0) if init == 'random' else gamma_mean(alpha=alpha0, beta=beta0) )
    return lamb

def initialise_U_gaussian(init, I, K, lamb):
    """ Initialise U, with prior Ui ~ N(0,I/lamb). """
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        lambdax = lamb[k] if isinstance(lamb, numpy.ndarray) else lamb
        U[i,k] = ( normal_draw(mu=0, tau=lambdax) if init == 'random' else normal_mean(mu=0, tau=lambdax) )
    return U

def initialise_U_exponential(init, I, K, lamb):
    """ Initialise U, with prior Uik ~ Exp(lamb). """
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        lambdax = lamb[k] if isinstance(lamb, numpy.ndarray) else lamb
        U[i,k] = ( exponential_draw(lambdax=lambdax) if init == 'random' else exponential_mean(lambdax=lambdax) )
    return U

def initialise_U_truncatednormal(init, I, K, mu, tau):
    """ Initialise U, with prior Uik ~ TruncatedNormal(mu,tau). """
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        U[i,k] = ( truncated_normal_draw(mu=mu, tau=tau) if init == 'random' else truncated_normal_mean(mu=mu, tau=tau) )
    return U

def initialise_U_halfnormal(init, I, K, sigma):
    """ Initialise U, with prior Uik ~ HalfNormal(sigma). """
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        U[i,k] = ( half_normal_draw(sigma=sigma) if init == 'random' else half_normal_mean(sigma=sigma) )
    return U

#TODO: many more