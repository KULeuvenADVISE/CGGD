# Constraint Guided Gradient Descent: Guided Training with Inequality Constraints with Applications in Regression and Semantic Segmentation


This is the code repository for the experiments described in the paper _Constraint Guided Gradient Descent: Guided Training with Inequality Constraints, Quinten Van Baelen, Peter Karsmakers_ (available [here](https://doi.org/10.14428/esann/2022.ES2022-105)), and _Constraint Guided Gradient Descent: Guided Training with Inequality Constraints with Applications in Regression and Semantic Segmentation, Quinten Van Baelen, Peter Karsmakers_ (available [here](https://doi.org/10.1016/j.neucom.2023.126636)).



The code for the semantic segmentation experiments, as described in [2], is based upon [Sizeloss_WSS](https://github.com/LIVIAETS/SizeLoss_WSS/tree/master) and [extended_logbarrier](https://github.com/LIVIAETS/extended_logbarrier/tree/master) because semantic segmentation tasks are very similar to the ones done in the corresponding papers. In each script it is clearly stated what has been added in this repository and what already existed in either of the previously mentioned repositories. The full possibility of the code as presented in these repositories are not guaranteed in this repository. This code base will most likely only support the experiments described in [2].

## How to run an experiment

Please look into the instructions described in the **README.md** file in the corresponding directory. Running the experiments in [1] is very similar to running the regression experiments in [2, Section 5.1], while running the experiments in [2, Section 5.2] is different.



## Content of directories
- [regression](https://github.com/KULeuvenADVISE/CGGD/tree/main/regression):
    - Contains the scripts necessary to perform the experiments described in [1].
- [regression-sup-semisup](https://github.com/KULeuvenADVISE/CGGD/tree/main/regression-sup-semisup):
    - Contains the scripts necessary to perform the experiments described in [2, Section 5.1] and a notebook to postprocess the results and visualize them in a similar fashion as was done in the corresponding paper.
- [semanticsegmentation](https://github.com/KULeuvenADVISE/CGGD/tree/main/semanticsegmentation):
    - Contains the scripts necessary to perform the experiments described in [2, Section 5.2] and a notebook to postprocess the results and visualize them in a similar fashion as was done in the corresponding paper.
- [venv-requirements](https://github.com/KULeuvenADVISE/CGGD/tree/main/venv-requirements):
    - Contains the virtual environments for running the experiments. This repository uses 3 different virtual environments. The first one ([requirements-regression](https://github.com/KULeuvenADVISE/CGGD/blob/main/venv-requirements/requirements-regression.txt)) is the virtual environment used for running the experiments in [**regression**](https://github.com/KULeuvenADVISE/CGGD/tree/main/regression). Note that this uses TensorFlow 2.2.0 and, thus, this is not supported for all GPUs (NVIDIA 30 series does not support this version of TensorFlow). The second one ([requirements-regression-sup-semisup.yml](https://github.com/KULeuvenADVISE/CGGD/blob/main/venv-requirements/requirements-regression-sup-semisup.yml)) contains the virtual environment for running the experiments in [**regression-sup-semisup**](https://github.com/KULeuvenADVISE/CGGD/tree/main/regression-sup-semisup). The third one ([requirements-semanticsegmentatio.ymln](https://github.com/KULeuvenADVISE/CGGD/blob/main/venv-requirements/requirements-semanticsegmentation.yml)) is the virtual environment used for running the experiments in [**semanticsegmentation**](https://github.com/KULeuvenADVISE/CGGD/tree/main/semanticsegmentation). This uses PyTorch because the implementation of the baselines was already available in PyTorch.



## Data Sets

This repository uses three publicly available data sets:
- The Bias Correction data set is available [here](https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast). For running the experiments on this data set, please download it and put it in __./regressions/DataSets__. Make sure that the file is named __Bias_correction_ucl.csv__, but normally this should be satisfied automatically.
- The Family Income data set is available [here](https://www.kaggle.com/grosvenpaul/family-income-and-expenditure).  For running the experiments on this data set, please download it and put it in __./regressions/DataSets__. Make sure that the file is named __Family Income and Expenditure.csv__, but normally this should be satisfied automatically.
- The Prostate data set is available [here](https://promise12.grand-challenge.org/Download/). Only the training data is used in the experiments, so it is only necessary to have __TrainingDataPart1.zip__, __TrainingDataPart2.zip__, and __TrainingDataPart3.zip__ downloaded from [here](https://zenodo.org/record/8014041).

The toy data set for the semantic segmentation experiments is created by running a script provided in the corresponding directory.

## References

[1] [_Constraint Guided Gradient Descent: Guided Training with Inequality Constraints, Quinten Van Baelen, Peter Karsmakers_](https://doi.org/10.14428/esann/2022.ES2022-105)

[2] [_Constraint Guided Gradient Descent: Guided Training with Inequality Constraints with Applications in Regression and Semantic Segmentation, Quinten Van Baelen, Peter Karsmakers_](https://doi.org/10.1016/j.neucom.2023.126636)

## Citation

If you use this repository, please cite one of the relevant papers. The citations can be found below.

```
@inproceedings{VanBaelen2022,
    author = {Van Baelen, Quinten and Karsmakers, Peter},
    booktitle = {Proceedings of the 30th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning},
    title = {Constraint guided gradient descent: Guided training with inequality constraints},
    year = {2022},
    doi = {10.14428/esann/2022.ES2022-105},
    pages = {175-180},
}

@article{VanBaelen2023,
    author = {Van Baelen, Quinten and Karsmakers, Peter},
    journal = {Neurocomputing},
    title = {Constraint Guided Gradiend Descent: Guided Training with Inequality Constraints with Applications in Regression and Semantic Segmentation},
    year = {2023},
    doi = {10.1016/j.neucom.2023.126636},
}
```

## Contact

For questions, problems and/or remarks, please contact the author: <quinten.vanbaelen@kuleuven.be> ORCID iD 0000-0003-2863-4227.

