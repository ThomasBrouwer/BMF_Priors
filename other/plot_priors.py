'''
Plot the following distributions, all with lambda = 0.1:
- Gaussian, N(0, lambda^-1)
- Exponential, Exp(lambda)
- Truncated normal, TN(0, lambda)
- Gamma, G(a,b)
(- Half Normal, HN(1/lambda))
'''

from scipy.stats import expon
#from scipy.stats import halfnorm
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import gamma
from scipy.stats import laplace
import numpy
import math
import matplotlib.pyplot as plt


plot_name = 'priors.png'
lamb = 0.1
sigma = 1./math.sqrt(lamb)
eta = sigma
a, b = 1., 1.


min_x, max_x, step = -10, 15, 0.01
min_y, max_y = 0., 0.35
x = numpy.arange(min_x, max_x+step, step)

fig = plt.figure(figsize=(6, 1.5))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.13, top=0.97)
plt.plot(x, norm.pdf(x, loc=0, scale=sigma), label='Normal')
plt.plot(x, laplace.pdf(x, loc=0, scale=eta), label='Laplace')
plt.plot(x, expon.pdf(x, scale=1./lamb), label='Exponential')
plt.plot(x, truncnorm.pdf(x, a=0, b=numpy.inf, loc=0., scale=sigma), label='Truncated normal')
plt.plot(x, gamma.pdf(x, a=a, scale=1./b), label='Gamma')
plt.legend(loc=1, fontsize=8)

plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
plt.xticks(fontsize=8)
plt.yticks([], fontsize=8)

plt.savefig(plot_name, dpi=600)


''' Print mean and variance. '''
print "Normal. Mean: %s. Variance: %s." % (
    norm.mean(loc=0, scale=sigma), norm.var(loc=0, scale=sigma))
print "Laplace. Mean: %s. Variance: %s." % (
    laplace.mean(loc=0, scale=eta), laplace.var(loc=0, scale=eta))
print "Exponential. Mean: %s. Variance: %s." % (
    expon.mean(scale=1./lamb), expon.var(scale=1./lamb))
print "Truncated normal. Mean: %s. Variance: %s." % (
    truncnorm.mean(a=0, b=numpy.inf, loc=0., scale=sigma), truncnorm.var(a=0, b=numpy.inf, loc=0., scale=sigma))
print "Gamma. Mean: %s. Variance: %s." % (
    gamma.mean(a=a, scale=1./b), gamma.var(a=a, scale=1./b))

