# Constraint Guided Gradient Descent: Guided Training with Inequality Constraints with Applications in Regression and Semantic Segmentation


The MakeFiles require that the main directory is a Git repo. So set current working directory to the directory with all the files (such as main and the MakeFiles). Then run 
```
>>> git innit
>>> git add .
```

Normally, the MakeFiles should unzip the data automatically and do this only when the data has not been unzipped before. However, it is possible that this does not work. A work around would be to manually unzip them. Make sure that when the data is unzipped, the filestructure looks like this:

```
+-- prostate
|   +-- .git
|   +-- data
|       +-- prostate
|       +-- promise
|       +-- TrainingData_Part1.zip
|       +-- TrainingData_Part2.zip
|       +-- TrainingData_Part3.zip
|       +-- prostate_v2.lineage
|   +-- models
|   +-- preprocess
|   +-- results
```


On Mac OS it could occur that an additional directory is created called *__Mac OS X__*. This directory should be removed. If the unzipping is done by the user, then the resulting file structure should be like this:
```
+-- prostate
|   +-- .git
|   +-- data
|       +-- prostate
|       +-- promise
|           +-- TrainingData_Part1
|           +-- TrainingData_Part2
|           +-- TrainingData_Part3
|       +-- TrainingData_Part1.zip
|       +-- TrainingData_Part2.zip
|       +-- TrainingData_Part3.zip
|       +-- prostate_v2.lineage
|   +-- models
|   +-- preprocess
|   +-- results
```

Afterwards, the MakeFiles should be able to skip the unzipping and perform the preprocessing and training.

The code for running the experiments for 50% and 10% of the training set works only when the data has been preprocessed first for the whole data set. The validation set is the same for each amount of trainig data. The creation of the smaller training sets is done by running the script __SampleTrainingSet.py__.


The advice order of running the MakeFiles is:
1. prostate.make,
2. prostate50.make,
3. prostate10.make.


After running all the MakeFiles, the results can be visualized using the __Results.ipynb__ notebook.


