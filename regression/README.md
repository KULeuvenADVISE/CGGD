# Constraint Guided Gradient Descent: Guided Training with Inequality Constraints

This directory contains the files required to rerun the experiments performed in _Constraint Guided Gradient Descent: Training with Inequality Constraints, Quinten Van Baelen, Peter Karsmakers_. The paper is available [here](https://www.esann.org/sites/default/files/proceedings/2022/ES2022-105.pdf). Note that the data sets themselves should be downloaded manually compatible with the instructions described in the Data Sets section below.


Important Packages and Versions
---
Please make sure that this section is compatible with the Python environment used to reproduce the experiments.

The code was ran and tested for the following versions of the packages listed below. All packages are necessary for running the provided scripts.

| Package | Version |
|---------| ------- | 
| numpy | 1.19.4 |
| pandas | 1.1.5 |
| scipy | 1.4.1 |
| scikit-learn | 0.20.2 |
| tensorflow | 2.2.0 |


For the full virtual environment that was used to perform these experiments, please install the environment as listed in [**venv-requirements/requirements-regression**](https://github.com/KULeuvenADVISE/CGGD/blob/main/venv-requirements/requirements-regression.txt). 


Data Sets
---

Before running any experiments, make sure that the data sets are downloaded and saved in the **DataSets** directory with the names __Bias_correction_ucl.csv__ and __Family Income and Expenditure.csv__. This should match the original filenames of the downloads.

Afterwards, one should run the script [__./DataSets/dataPreprocessing.py__](https://https://github.com/KULeuvenADVISE/CGGD/blob/main/regression/DataSets/dataPreprocessing.py) in order to construct the data samples that are used to perform the experiments. More explanation about the sampling is given in the README file in the same directory.  

For convenience:
- The Bias Correction data set is available [here](https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast) and in the directory [**DataSets**](https://https://github.com/KULeuvenADVISE/CGGD/blob/main/regression/DataSets).
- The Family Income data set is available [here](https://www.kaggle.com/grosvenpaul/family-income-and-expenditure) and in the directory [**DataSets**](https://https://github.com/KULeuvenADVISE/CGGD/blob/main/regression/DataSets).




Hypothesis 1
---

The performance of the model trained by **CGGD** is less dependent on the initialization of the network weights compared to a model trained without constraints or a model trained with a fuzzy-like loss function. More specific, the standard deviation of the performance of the models obtained by applying **CGGD** is lower than that of the models obtained from unconstrained training and the mean is at least comparable between the models from both methods.

The files used to perform the experiments to check this hypothesis are in the directory _Hypothesis1_.


