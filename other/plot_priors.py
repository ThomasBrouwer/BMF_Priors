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
import numpy
import math
import matplotlib.pyplot as plt


plot_name = 'priors.png'
lamb = 0.1
sigma = 1./math.sqrt(lamb)


min_x, max_x, step = -10, 25, 0.01
min_y, max_y = 0., 0.35
x = numpy.arange(min_x, max_x+step, step)

fig = plt.figure(figsize=(6, 1.5))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.13, top=0.97)
plt.plot(x, norm.pdf(x, loc=0, scale=sigma), label='Normal')
#plt.plot(x, halfnorm.pdf(x, loc=0, scale=sigma), label='Half normal')
plt.plot(x, expon.pdf(x, scale=1./lamb), label='Exponential')
plt.plot(x, truncnorm.pdf(x, a=0, b=numpy.inf, loc=0., scale=sigma), label='Truncated normal')
plt.plot(x, gamma.pdf(x, a=1, scale=1), label='Gamma')
plt.legend(loc=1, fontsize=10)

plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
plt.xticks(fontsize=8)
plt.yticks([], fontsize=8)

plt.savefig(plot_name, dpi=600)