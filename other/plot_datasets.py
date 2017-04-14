'''
Plot the distributions of the drug sensitivity, MovieLens, and Jester datasets.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.data.drug_sensitivity.load_data import load_ctrp_ec50_integer
from BMF_Priors.data.drug_sensitivity.load_data import load_ccle_ic50_integer
from BMF_Priors.data.drug_sensitivity.load_data import load_ccle_ec50_integer
from BMF_Priors.data.movielens.load_data import load_movielens_100K
from BMF_Priors.data.movielens.load_data import load_movielens_1M
from BMF_Priors.data.jester.load_data import load_jester_data_integer

import itertools
import matplotlib.pyplot as plt


''' Load the data. '''
def extract_values(R, M):
    I, J = R.shape
    return [R[i,j] for i,j in itertools.product(range(I),range(J)) if M[i,j]]
    
#R_gdsc,    M_gdsc =    load_gdsc_ic50_integer()
#R_ctrp,    M_ctrp =    load_ctrp_ec50_integer()
#R_ccle_ic, M_ccle_ic = load_ccle_ic50_integer()
#R_ccle_ec, M_ccle_ec = load_ccle_ec50_integer()
#R_100K,    M_100K =    load_movielens_100K()
#R_1M,      M_1M =      load_movielens_1M()
R_jester,  M_jester =  load_jester_data_integer()

values_plotnames_bins = [
#    (extract_values(R_gdsc, M_gdsc), 'plot_gdsc.png', [v-0.5 for v in range(0,100+10,5)]),
#    (extract_values(R_ctrp, M_ctrp), 'plot_ctrp.png', [v-0.5 for v in range(0,100+10,5)]),
#    (extract_values(R_ccle_ic, M_ccle_ic), 'plot_ccle_ic.png', [v-0.5 for v in range(0,8+2)]),
#    (extract_values(R_ccle_ec, M_ccle_ec), 'plot_ccle_ec.png', [v-0.5 for v in range(0,10+2)]),
#    (extract_values(R_100K, M_100K), 'plot_movielens_100k.png', [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
 #   (extract_values(R_1M, M_1M), 'plot_movielens_1m.png', [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
    (extract_values(R_jester, M_jester), 'plot_jester.png', [v-0.5 for v in range(1,20+2)]),
]


''' Make the plots. '''
for values, plotname, bins in values_plotnames_bins:
    fig = plt.figure(figsize=(2, 1))
    fig.subplots_adjust(left=0.03, right=0.95, bottom=0.15, top=0.99)
    plt.hist(values, bins=bins)
    
    plt.xticks(fontsize=8)
    plt.yticks([], fontsize=8)
    plt.xlim(bins[0], bins[-1])
    
    plt.savefig(plotname, dpi=600)