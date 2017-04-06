'''
Plot the following distributions, all with lambda = 0.1:
- Gaussian, N(0, lambda^-1)
- Exponential, Exp(lambda)
- Truncated normal, TN(0, lambda)
- Half Normal, HN(1/lambda)
'''

from scipy.stats import expon
from scipy.stats import halfnorm
from scipy.stats import norm
from scipy.stats import truncnorm
import numpy
import math
import matplotlib.pyplot as plt


plot_name = 'priors.png'
lamb = 0.1
sigma = 1./math.sqrt(lamb)


min_x, max_x, step = -10, 30, 0.01
min_y, max_y = 0., 0.3
x = numpy.arange(min_x, max_x+step, step)

fig = plt.figure(figsize=(6, 2))
fig.subplots_adjust(left=0.05, right=0.97, bottom=0.10, top=0.97)
plt.plot(x, norm.pdf(x, loc=0, scale=sigma), label='Normal')
plt.plot(x, halfnorm.pdf(x, loc=0, scale=1./lamb), label='Half normal')
plt.plot(x, expon.pdf(x, scale=1./lamb), label='Exponential')
plt.plot(x, truncnorm.pdf(x, a=0, b=numpy.inf, loc=0., scale=sigma), label='Truncated normal')
plt.legend(loc=1, fontsize=10)

plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.savefig(plot_name, dpi=600)