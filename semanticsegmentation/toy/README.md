# Constraint Guided Gradient Descent: Guided Training with Inequality Constraints with Applications in Regression and Semantic Segmentation


The MakeFiles require that the main directory is a Git repo. So set current working directory to the directory with all the files (such as main and the MakeFiles). Then run 
```
>>> git innit
>>> git add .
```

The data will be automatically generated by the MakeFile __toy.make__ if it does not exist yet. First, the full data set should be ran before running the experiment with 50% and 10% of the training data because they use a sample of the data generated by the training set. To generate these sampled data sets, simply run the file __SampleTrainingSet.py__. If the directory contains the files __all_files_50.pickle__ and/or __all_files_10.pickle__, then this functio will sample exactly the images listed in these files. If either has been removed, then it will generate randomly new images from the training set. The validation set will be copied and the same for each setting.

The advice order of running the MakeFiles is:
1. toy.make,
2. toy50.make,
3. toy10.make.


After running all the MakeFiles, the results can be visualized using the __Results.ipynb__ notebook.


