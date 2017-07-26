'''
Measure convergence on the GDSC drug sensitivity dataset, with the column-average 
baseline (i.e. simply the training error repeated for each iteration).
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.baseline_average_column import ColumnAverage
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer

import matplotlib.pyplot as plt


''' Train the model and repeat the training error :iterations times. '''
R, M = load_gdsc_ic50_integer()
model_class = ColumnAverage
iterations = 200
fout_performances = './results/performances_baseline_average_column.txt'

BMF = model_class(R=R,M=M,K=0,hyperparameters={}) 
BMF.initialise(init='')
BMF.run(iterations)

training_error = BMF.predict(M_pred=M,burn_in=0,thinning=0)['MSE']
performances = [training_error for i in range(iterations)]
open(fout_performances,'w').write("%s" % performances)


''' Plot the performance vs iterations. '''
plt.figure()
plt.title("Performance against iteration")
plt.plot(performances)
plt.ylim(0,2000)
