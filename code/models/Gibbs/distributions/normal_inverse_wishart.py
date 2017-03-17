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

# Draw a value for mu, Sigma ~ NIW(mu0,beta0,v0,W0)
def normal_inverse_wishart_draw(mu0,beta0,v0,W0):
    Sigma = invwishart.rvs(df=v0, scale=W0)
    mu = multivariate_normal_draw(mu=mu0,sigma=Sigma/beta0)
    return (mu,Sigma)
    
       
'''
# Example draw
I = 5
mu0, beta0, v0, W0 = numpy.zeros(I), 1., I, numpy.eye(I)
print normal_inverse_wishart_draw(mu0,beta0,v0,W0)
'''