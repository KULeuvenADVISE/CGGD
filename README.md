# Constraint Guided Gradient Descent: Guided Training with Inequality Constraints

This is the code repository for the experiments described in the paper _Constraint Guided Gradient Descent: Training with Inequality Constraints, Quinten Van Baelen, Peter Karsmakers_. The paper is available [here](https://www.esann.org/sites/default/files/proceedings/2022/ES2022-105.pdf).

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


How to run any file
---

When running a certain file it is required to set the current working directory to the directory of the file that is run. Otherwise, the paths are not constructed properly and the files will not work. For example, if the current working directory is set to the directory of this file, then the experiments for the bias correction data set can be run by:

```
$ cd ./Datasets/
$ python dataPreprocessing.py

$ cd ../Hypothesis1/BiasCorrection/
$ python main.py
```

When the data preprocessing is already performed (and the current working directory is set to the directory of this file), then the experiments of the bias correction data set can be run by:

```
$ cd ./Hypothesis1/BiasCorrection/
$ python main.py
```

Data Sets
---

Before running any experiments, make sure that the data sets are downloaded and saved in the **DataSets** directory with the names __Bias_correction_ucl.csv__ and __Family Income and Expenditure.csv__. This should match the original filenames of the downloads.

Afterwards, one should run the script __./DataSets/dataPreprocessing.py__ in order to construct the data samples that are used to perform the experiments. More explanation about the sampling is given in the README file in the same directory.  

For convenience:
- The Bias Correction data set is available [here](https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast) and in the directory **DataSets**.
- The Family Income data set is available [here](https://www.kaggle.com/grosvenpaul/family-income-and-expenditure) and in the directory **DataSets**.




Hypothesis 1
---

The performance of the model trained by **CGGD** is less dependent on the initialization of the network weights compared to a model trained without constraints or a model trained with a fuzzy-like loss function. More specific, the standard deviation of the performance of the models obtained by applying **CGGD** is lower than that of the models obtained from unconstrained training and the mean is at least comparable between the models from both methods.

The files used to perform the experiments to check this hypothesis are in the directory _Hypothesis1_.


Contact
---
For questions, problems and/or remarks, please contact the author: <quinten.vanbaelen@kuleuven.be> ORCID iD 0000-0003-2863-4227.

Reference
---
Please use the citation below if you want to refer to this work.

~~~
@inproceedings{VanBaelen2022CGGD,
    author = {Van Baelen, Quinten and Karsmakers, Peter},
    booktitle = {Proceedings of the 30th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning},
    title = {Constraint guided gradient descent: Guided training with inequality constraints},
    year = {2022},
    doi = {https://doi.org/10.14428/esann/2022.ES2022-105},
}
~~~

