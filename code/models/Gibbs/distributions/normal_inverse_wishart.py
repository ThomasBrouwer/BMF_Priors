"""
Class representing a Normal-Inverse Wishart distribution, allowing us to sample from it.

Say mu, Sigma ~ NIW(mu0,beta0,v0,W0).
To sample:
- Sample Sigma ~ IW(v0, W0)
- Sample mu ~ N(mu0, 1/beta0 * W0)

https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution#Generating_normal-inverse-Wishart_random_variates
"""
from scipy.stats import invwishart
from multivariate_normal import multivariate_normal_draw
import numpy

# Draw a value for mu, Sigma ~ NIW(mu0,beta0,v0,W0)
def normal_inverse_wishart_draw(mu0,beta0,v0,W0):
    sigma = invwishart.rvs(df=v0, scale=W0)
    sigma = sigma if sigma.shape != () else numpy.array([[sigma]])
    mu = multivariate_normal_draw(mu=mu0,sigma=sigma/beta0)
    return (mu,sigma)
    
def normal_inverse_wishart_mean(mu0,beta0,v0,W0):
    # Mean of InverseWishart is W0 / ( v0 - K - 1 ). Mean of Normal is mu0.
    K = mu0.shape[0]
    return (mu0, W0 / (v0 - K - 1))
       
'''
# Example draw
I = 5
mu0, beta0, v0, W0 = numpy.zeros(I), 1., I, numpy.eye(I)
print normal_inverse_wishart_draw(mu0,beta0,v0,W0)
'''