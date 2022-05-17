Data Preprocessing
===

The preprocessing on the datasets is done by the code in the script **dataPreprocessing.py**. 

For the Bias Correction data set, the preprocessing consists of splitting up the date of the recording to three columns day-month-year such that the variables become integers, normalizing all the input and output values, and taking a sample of 2500 examples out of the whole data set.

For the Family Income data set, the preprocessing consists of removing the examples that do not satisfy the income constraint (this is a very small number of examples), extracting the variables used in the experiment, normalizing the input and output values, and taking a sample of 2500 examples out of the whole data set.

