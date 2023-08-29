Data Preprocessing
===

The preprocessing on the datasets is done by the code in the script **dataPreprocessing.py**. This script can be ran as long as both data sets are provided in this directory. The script will by default preprocess both data sets at once. By adjusting the values of __run_bc__ (line 13) and __run_fi__ (line 14), one can choose to only preprocess a certain data set. Setting the value to __False__ will make the code skip the preprocessing. This can be useful if you only download one data set. The variable __run_bc__ corresponds to the Bias Correction data set and the variable __run_fi__ corresponds to the Family Income data set.

For the Bias Correction data set, the preprocessing consists of splitting up the date of the recording to three columns day-month-year such that the variables become integers, normalizing all the input and output values, and taking a sample of 2500 examples out of the whole data set.

For the Family Income data set, the preprocessing consists of removing the examples that do not satisfy the income constraint (this is a very small number of examples), extracting the variables used in the experiment, normalizing the input and output values, and taking a sample of 2500 examples out of the whole data set. The exact number of examples that are removed by adding the income constraint can be found by comparing the number of rows of the dataframe before and after running line 231 of the script.

