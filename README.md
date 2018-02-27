# Prior and Likelihood Choices for Bayesian Matrix Factorisation on Small Datasets

This project contains implementations of sixteen Bayesian matrix factorisation models studied in the paper [**Prior and Likelihood Choices for Bayesian Matrix Factorisation on Small Datasets**](https://arxiv.org/abs/1712.00288). We furthermore provide all datasets used (including the preprocessing scripts), and Python scripts for experiments.

#### Paper abstract
In this paper, we study the effects of different prior and likelihood choices for Bayesian matrix factorisation, focusing on small datasets. These choices can greatly influence the predictive performance of the methods. We identify four groups of approaches: Gaussian-likelihood with real-valued priors, nonnegative priors, semi-nonnegative models, and finally Poisson-likelihood approaches. For each group we review several models from the literature, considering sixteen in total, and discuss the relations between different priors and matrix norms. We extensively compare these methods on eight real-world datasets across three application areas, giving both inter- and intra-group comparisons. We measure convergence runtime speed, cross-validation performance, sparse and noisy prediction performance, and model selection robustness. We offer several insights into the trade-offs between prior and likelihood choices for Bayesian matrix factorisation on small datasets - such as that Poisson models give poor predictions, and that nonnegative models are more constrained than real-valued ones.

#### Authors
Thomas Brouwer, Pietro Lio'. 
Contact: tab43@cam.ac.uk / thomas.a.brouwer@gmail.com.

## Installation 
If you wish to use the matrix factorisation models, or replicate the experiments, follow these steps. Please ensure you have Python 2.7 (3 is currently not supported). 
1. Clone the project to your computer, by running `git clone https://github.com/ThomasBrouwer/BMF_Priors.git` in your command line.
2. In your Python script, add the project to your system path using the following lines.  
   
   ``` 
   project_location = "/path/to/folder/containing/project/"
   import sys
   sys.path.append(project_location) 
   ```
   For example, if the path to the project is `/johndoe/projects/BMF_Priors/`, use `project_location = /johndoe/projects/`. 
   If you intend to rerun some of the paper's experiments, those scripts automatically add the correct path.
3. You may also need to add an empty file in `/johndoe/projects/` called `__init__.py`.
4. You can now import the models in your code, e.g.
```
from BMF_Priors.code.models.bmf_gaussian_exponential.py import BMF_Gaussian_Exponential
import numpy
R, M = numpy.ones((4,3)), numpy.ones((4,3))
model = BMF_Gaussian_Exponential(R=R, M=M, K=2, hyperparameters={})
model.initialise()
model.train(iterations=10)
model.predict(M_pred=M,burn_in=5,thinning=1)
```

## Examples
You can find good examples of the models running on data in the [convergence experiment](./experiments/convergence/convergence_experiment.py), e.g. [the model with exponential priors and a Gaussian likelihood](./experiments/convergence/drug_sensitivity_gdsc/gaussian_exponential.py).

## Citation
If this project was useful for your research, please consider citing our [arXiv paper](https://arxiv.org/abs/1712.00288).
> Thomas Brouwer and Pietro Lió (2017). Prior and Likelihood Choices for Bayesian Matrix Factorisation on Small Datasets. arXiv preprint arXiv:1712.00288.
```
@article{Brouwer2017c,
  title={{Prior and Likelihood Choices for Bayesian Matrix Factorisation on Small Datasets}},
  author={Brouwer, Thomas and Lio, Pietro},
  journal={arXiv preprint arXiv:1712.00288},
  year={2017}
}
```

## Project structure
<details>
<summary>Click here to find a description of the different folders and files available in this repository.</summary>

<br>

### /code/
Python code, for the models and cross-validation methods.

**/models/**: the Gibbs sampling code to compute the parameter values and update the random variables, as well as the matrix factorisation models that use them.
- **/Gibbs/distributions/** - Folder containing wrappers for the different probability distributions we use (exponential, Gaussian, normal-inverse Wishart, ..).
- **/Gibbs/parameters.py** - Methods returning the parameter values for each of the models.
- **/Gibbs/updates.py** - Methods for drawing new values for the variables (effectively implementing the Gibbs sampler for each variable).
- **/Gibbs/initialise.py** - Methods for initialising the random variables, either using the expectation of the priors, or using random draws (in the paper we use random draws).
- **bmf.py** - The general class for the Bayesian matrix factorisation methods. All other classes extend this one, and implement the specific models presented in the paper.
- **bmf_gaussian_gaussian.py** - All Gaussian model (GGG).
- **bmf_gaussian_gaussian_univariate.py** - All Gaussian model with univariate posterior (GGGU).
- **bmf_gaussian_gaussian_ard.py** - All Gaussian model with ARD hierarchical prior (GGGA).
- **bmf_gaussian_gaussian_wishart.py** - All Gaussian model with Wishart hierarchical prior (GGGW).
- **bmf_gaussian_laplace.py** - Gaussian likelihood with Laplace priors (GLL).
- **bmf_gaussian_laplace_ig.py** - Gaussian likelihood with Laplace priors and hierarchical Inverse Gaussian prior.
- **bmf_gaussian_gaussian_volumeprior.py** - Gaussian likelihood with volume prior (GVG).
- **bmf_gaussian_exponential.py** - Gaussian likelihood with exponential priors (GEE).
- **bmf_gaussian_exponential_ard.py** - Gaussian likelihood with exponential prior and ARD (GEEA).
- **bmf_gaussian_truncatednormal.py** - Gaussian likelihood with truncated normal priors (GTT).
- **bmf_gaussian_truncatednormal_hierarchical.py** - Gaussian likelihood with truncated normal and hierarchical priors (GTTN).
- **bmf_gaussian_l21.py** - Gaussian likelihood with L21 priors (GL21).
- **bmf_gaussian_gaussian_volumeprior_nonnegative.py** - Gaussian likelihood with nonnegative volume prior (GVnG).
- **bmf_gaussian_gaussian_exponential.py** - Gaussian likelihood with exponential and Gaussian priors (GEG).
- **bmf_poisson_gamma.py** - Poisson likelihood with Gamma priors (PGG).
- **bmf_poisson_gamma_gamma.py** - Poisson likelihood with Gamma and hierarchical Gamma priors  (PGGG).
- **baseline_mf_nonprobabilistic.py** - Nonprobabilistic MF model for baseline comparison (based on I-divergence model in Lee and Seung 2000).
- **baseline_average_row.py** - Baseline model that uses the row average for predicting missing values.
- **baseline_average_column.py** - Baseline model that uses the column average for predicting missing values.

**/cross_validation/**: Classes for doing cross-validation, and nested cross-validation, on the Bayesian NMF and NMTF models
- **matrix_cross_validation.py** - Class for finding the best value of K for any of the models (Gibbs, VB, ICM, NP), using cross-validation.
- **parallel_matrix_cross_validation.py** - Same as matrix_cross_validation.py, but P folds are ran in parallel (not used).
- **nested_matrix_cross_validation.py** - Class for measuring cross-validation performance, with nested cross-validation to choose K.
- **matrix_single_cross_validation.py** - Class for measuring cross-validation performance, for a specified K (no nested cross-validation).
- **mask.py** - Contains methods for splitting data into training and test folds (also used in model selection and sparsity experiments).

### /data/
Folder containing the datasets used.
- **/drug_sensitivity/** - Data and code for loading the drug sensitivity data. Methods for loading the data are provided in **load_data.py**; more details are available in **description.md**. The datasets are available in raw and post-processing format (**/raw/**, **/processed_all/**), together with the processing scripts (**process_raw_gdsc.py**, **process_raw_ctrp.py**, **process_raw_ccle.py**).
- **/movielens/** - Data and code for loading the MovieLens 100K data. The 1M dataset is also provided. Methods for loading the data are provided in **load_data.py**; more details are available in **description.md**.
- **/methylation/** - Data and code for loading the gene body and promoter region methylation data. Methods for loading the data are provided in **load_data.py**; more details are available in **description.md**.
- We also provide the Jester joke rating dataset (**/jester/**), and two cancer methylation datasets (**/methylation/**), although we did not use them.

### /experiments/
Scripts for the experiments, along with the results and plots thereof.
- **/convergence/** - Measure convergence rate of the methods (against iterations and time) on the GDSC and MovieLens 100K datasets.
- **/runtime/** - Measure the time taken per iteration for different values of K, on the GDSC and MovieLens 100K datasets.
- **/cross_validation/** - 5-fold cross-validation experiments on the eight datasets.
- **/sparsity/** - Measure the predictive performance on missing values for varying sparsity levels on the GDSC, MovieLens 100K, and gene body methylation datasets.
- **/noise/** - Measure the predictive performance on missing values for varying noise levels on the GDSC, MovieLens 100K, and gene body methylation datasets.
- **/model_selection/** - Measure the predictive performance on missing values for different values of K, on the GDSC, MovieLens 100K, and gene body methylation datasets.
- **/factor_analysis/** - Scripts for storing the factor values on the GDSC dataset, as well as scripts for plotting those factor values.
- **/parameter_exploration/** - Scripts for trying different values of K for the volume prior models, on each of the datasets.

### /other/
Miscellaneous scripts for plotting the datasets, legends, and priors.

</br>
</details>
