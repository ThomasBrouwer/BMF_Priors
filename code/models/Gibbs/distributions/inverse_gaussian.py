"""
Class representing an Inverse Gaussian distribution, allowing us to sample from it.

x ~ IG(mu, tau) = (tau / (2*pi*x^3))^1/2 * exp{ -lambda * (x-mu)^2 / ( 2 * mu^2 * x }
"""
from numpy.random import wald

def inverse_gaussian_draw(mu,tau):
    return wald(mean=mu, scale=tau, size=None)
    
def inverse_gaussian_mean(mu,tau):
    return mu    
    
       
'''
# Do 1000 draws and plot them
import matplotlib.pyplot as plt
import numpy as np
mu = 1.
tau = 4.
s = [inverse_gaussian_draw(mu,tau) for i in range(0,1000)] 
s2 = np.random.wald(mu,tau, 1000)
count, bins, ignored = plt.hist(s, 50, normed=True)
count, bins, ignored = plt.hist(s2, 50, normed=True)
plt.show()
'''