# Constraint Guided Gradient Descent: Guided Training with Inequality Constraints with Applications in Regression and Semantic Segmentation


This repository contains code to perform the experiments described in [2, Section 5.1]. The files with a name starting with __BiasCorrection__ correspond to the Temperature experiments ([2, Section 5.1.1]) and the files with a name starting with __FamilyIncome__ correspond to the Family income experiments [2, Section 5.1]. The details of the setups are described in the paper, but the usage of the files are completely the same for both data sets. Hence, only the files relevant for the Temperature experiment are discussed in detail.

## Different sizes

In order to run the experiment for the fully supervised experiment, run the file [__BiasCorrectionDiffSizeMain.py__](https://github.com/KULeuvenADVISE/CGGD/blob/main/regression-sup-semisup/comparison/BiasCorrectionDiffSizeMain.py). This can be done by first setting the current working directory to this one and simply running it, i.e. if we assume that the current workind directory is the root directory of this repository, then run the following in your terminal will perform the experiments.

```
>>> cd ./regression-sup-semisup/comparison/
>>> python BiasCorrectionDiffSizeMain.py
```

Each file contains documentation on what each function does, but the following structure is used:
- [__BiasCorrectionDiffSizeMain.py__](https://github.com/KULeuvenADVISE/CGGD/blob/main/regression-sup-semisup/comparison/BiasCorrectionDiffSizeMain.py) is the main file that allows the user to run the full experiment or part of it, for example by only running the training of CGGD or the unconstrained baseline. See the file for more information on how to do this.
- [__BiasCorrectionAux.py__](https://github.com/KULeuvenADVISE/CGGD/blob/main/regression-sup-semisup/comparison/BiasCorrectionAux.py) is the file that contains code to load the data, create the model architecture and loops over the different training settings for the different training methods.
- [__BiasCorrectionCGGD.py__](https://github.com/KULeuvenADVISE/CGGD/blob/main/regression-sup-semisup/comparison/BiasCorrectionCGGD.py) is the file that contains functions to train the model with each method as well as checking of the constraints, computing the direction of the constraints, computing the satisfaction ratio, and all standard steps of training neural networks.
- [__BiasCorrectionConstraints.py__](https://github.com/KULeuvenADVISE/CGGD/blob/main/regression-sup-semisup/comparison/BiasCorrectionConstraints.py) is the file that contains the definition of the bound constraints. Note that this should be adjusted for different data sets, because typically different bound constraints are considered.


At last, the script [__PostprocessDiffSize.py__](https://github.com/KULeuvenADVISE/CGGD/blob/main/regression-sup-semisup/comparison/PostprocessDiffSize.py) contains code to extract the relevant metrics and construct tables representing the results for the different sizes of training sets. Note that this file assumes that all methods have been used for training the models.

## Semi-supervised

In order to run the experiment for the fully supervised experiment, run the file [__BiasCorrectionSemiSupMain.py__](https://github.com/KULeuvenADVISE/CGGD/blob/main/regression-sup-semisup/comparison/BiasCorrectionSemiSupMain.py). This can be done by first setting the current working directory to this one and simply running it, i.e. if we assume that the current workind directory is the root directory of this repository, then run the following in your terminal will perform the experiments.

```
>>> cd ./regression-sup-semisup/comparison/
>>> python BiasCorrectionSemiSupMain.py
```

The script [__PostprocessSemiSup.py__](https://github.com/KULeuvenADVISE/CGGD/blob/main/regression-sup-semisup/comparison/PostprocessSemiSup.py) contains code to extract the relevant metrics and construct tables representing the results for the semi-supervised training sets. Note that this file assumes that all methods have been used for training the models.

