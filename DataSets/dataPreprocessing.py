"""
Script that performs the necessary preprocessing on the data set.

:author: Quinten Van Baelen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import numpy as np
import pandas as pd
import scipy.io as io


def date_to_datetime(df):
    """Transform the string that denotes the date to the datetime format in pandas."""
    # make copy of dataframe
    df_temp = df.copy()
    # add new column at the front where the date string is transformed to the datetime format
    df_temp.insert(0, 'DateTransformed', pd.to_datetime(df_temp['Date']))
    return df_temp


def add_year(df):
    """Extract the year from the datetime cell and add it as a new column to the dataframe at the front."""
    # make copy of dataframe
    df_temp = df.copy()
    # extract year and add new column at the front containing these numbers
    df_temp.insert(0, 'Year', df_temp['DateTransformed'].dt.year)
    return df_temp


def add_month(df):
    """Extract the month from the datetime cell and add it as a new column to the dataframe at the front."""
    # make copy of dataframe
    df_temp = df.copy()
    # extract month and add new column at index 1 containing these numbers
    df_temp.insert(1, 'Month', df_temp['DateTransformed'].dt.month)
    return df_temp


def add_day(df):
    """Extract the day from the datetime cell and add it as a new column to the dataframe at the front."""
    # make copy of dataframe
    df_temp = df.copy()
    # extract day and add new column at index 2 containing these numbers
    df_temp.insert(2, 'Day', df_temp['DateTransformed'].dt.day)
    return df_temp


def add_input_output_temperature(df):
    """Add a multiindex denoting if the column is an input or output variable."""
    # copy the dataframe
    temp_df = df.copy()
    # extract all the column names
    column_names = temp_df.columns.tolist()
    # only the last 2 columns are output variables, all others are input variables. So make list of corresponding lengths of 'Input' and 'Output'
    input_list = ['Input'] * (len(column_names) - 2)
    output_list = ['Output'] * 2
    # concat both lists
    input_output_list = input_list + output_list
    # define multi index for attaching this 'Input' and 'Output' list with the column names already existing
    multiindex_bias = pd.MultiIndex.from_arrays([input_output_list, column_names])
    # transpose such that index can be adjusted to multi index
    new_df = pd.DataFrame(df.transpose().to_numpy(), index=multiindex_bias)
    # transpose back such that columns are the same as before except with different labels
    return new_df.transpose()


def add_input_output_family_income(df):
    """Add a multiindex denoting if the column is an input or output variable."""
    # copy the dataframe
    temp_df = df.copy()
    # extract all the column names
    column_names = temp_df.columns.tolist()
    # the 2nd-9th columns correspond to output variables and all others to input variables. So make list of corresponding lengths of 'Input' and 'Output'
    input_list_start = ['Input']
    input_list_end = ['Input'] * (len(column_names) - 9)
    output_list = ['Output'] * 8
    # concat both lists
    input_output_list = input_list_start + output_list + input_list_end
    # define multi index for attaching this 'Input' and 'Output' list with the column names already existing
    multiindex_bias = pd.MultiIndex.from_arrays([input_output_list, column_names])
    # transpose such that index can be adjusted to multi index
    new_df = pd.DataFrame(df.transpose().to_numpy(), index=multiindex_bias)
    # transpose back such that columns are the same as before except with different labels
    return new_df.transpose()


def normalize_columns_bias(df):
    """Normalize the columns for the bias correction dataset. This is different from normalizing all the columns separately because the
    upper and lower bounds for the output variables are assumed to be the same."""
    # copy the dataframe
    temp_df = df.copy()
    # normalize each column
    for feature_name in df.columns:
        # the output columns are normalized using the same upper and lower bound for more efficient check of the inequality
        if feature_name == 'Next_Tmax' or feature_name == 'Next_Tmin':
            max_value = 38.9
            min_value = 11.3
        # the input columns are normalized using their respective upper and lower bounds
        else:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
        temp_df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return temp_df


def sample_2500_examples(df):
    """Sample 2500 examples from the dataframe without replacement."""
    temp_df = df.copy()
    sample_df = temp_df.sample(n=2500, replace=False, random_state=3, axis=0)
    return sample_df


def sample_5000_examples(df):
    """Sample 5000 examples from the dataframe without replacement."""
    temp_df = df.copy()
    sample_df = temp_df.sample(n=5000, replace=False, random_state=4, axis=0)
    return sample_df


bias_correction_reduced = (
    # load dataset
    pd.read_csv("Bias_correction_ucl.csv")
    # drop missing values
    .dropna(how='any')
    # transform string date to datetime format
    .pipe(date_to_datetime)
    # add year as a single column
    .pipe(add_year)
    # add month as a single column
    .pipe(add_month)
    # add day as a single column
    .pipe(add_day)
    # remove original date string and the datetime format
    .drop(['Date', 'DateTransformed'], axis=1, inplace=False)
    # convert all numbers to float32
    .astype('float32')
    # normalize columns
    .pipe(normalize_columns_bias)
    # add multi index indicating which columns are corresponding to input and output variables
    .pipe(add_input_output_temperature)
    # sample 5000 examples out of the dataset
    .pipe(sample_5000_examples)
    # transpose such that the multiindex is indeed an index and not a column
    .transpose()
)

bias_correction_reduced.to_csv('./Sample_Bias_correction_ucl_reduced.csv')


def normalize_columns_income(df):
    """Normalize the columns for the Family Income dataframe. This can also be applied to other dataframes because this function normalizes
    all columns individually."""
    # copy the dataframe
    temp_df = df.copy()
    # normalize each column
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        temp_df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return temp_df


def check_constraints_income(df):
    """Check if all the constraints are satisfied for the dataframe and remove the examples that do not satisfy the constraint. This
    function only works for the Family Income dataset and the constraints are that the household income is larger than all the expenses
    and the food expense is larger than the sum of the other (more detailed) food expenses."""
    temp_df = df.copy()
    # check that household income is larger than expenses in the output
    input_array = temp_df['Input'].to_numpy()
    income_array = np.add(np.multiply(input_array[:, [0, 1]],
                                      np.subtract(np.asarray([11815988, 9234485]),
                                                  np.asarray([11285, 0]))),
                          np.asarray([11285, 0]))
    expense_array = temp_df['Output'].to_numpy()
    expense_array = np.add(np.multiply(expense_array,
                                       np.subtract(np.asarray([791848, 437467, 140992, 74800, 2188560, 1049275, 149940, 731000]),
                                                   np.asarray([3704, 0, 0, 0, 1950, 0, 0, 0]))),
                           np.asarray([3704, 0, 0, 0, 1950, 0, 0, 0]))
    expense_array_without_dup = expense_array[:, [0, 4, 5, 6, 7]]
    sum_expenses = np.sum(expense_array_without_dup, axis=1)
    total_income = np.sum(income_array, axis=1)
    sanity_check_array = np.greater_equal(total_income, sum_expenses)
    temp_df['Unimportant'] = sanity_check_array.tolist()
    reduction = temp_df[temp_df.Unimportant]
    drop_reduction = reduction.drop('Unimportant', axis=1)

    # check that the food expense is larger than all the sub expenses
    expense_reduced_array = drop_reduction['Output'].to_numpy()
    expense_reduced_array = np.add(np.multiply(expense_reduced_array,
                                               np.subtract(np.asarray([791848, 437467, 140992, 74800, 2188560, 1049275, 149940, 731000]),
                                                           np.asarray([3704, 0, 0, 0, 1950, 0, 0, 0]))),
                                   np.asarray([3704, 0, 0, 0, 1950, 0, 0, 0]))
    food_mul_expense_array = expense_reduced_array[:, [1, 2, 3]]
    food_mul_expense_array_sum = np.sum(food_mul_expense_array, axis=1)
    food_expense_array = expense_reduced_array[:, 0]
    sanity_check_array = np.greater_equal(food_expense_array, food_mul_expense_array_sum)
    drop_reduction['Unimportant'] = sanity_check_array.tolist()
    new_reduction = drop_reduction[drop_reduction.Unimportant]
    satisfied_constraints_df = new_reduction.drop('Unimportant', axis=1)

    return satisfied_constraints_df


family_income_and_expenditure_reduced = (
    # read file
    pd.read_csv("Family Income and Expenditure.csv")
    # drop missing values
    .dropna(how='any')
    # convert object to fitting dtype
    .convert_dtypes()
    # remove all strings (no other dtypes are present except for integers and floats)
    .select_dtypes(exclude=['string'])
    # transform all numbers to same dtype
    .astype('float32')
    # drop column with label Agricultural Household indicator because this is not really a numerical input but rather a categorical/classification
    .drop(['Agricultural Household indicator'], axis=1, inplace=False)
    # this column is dropped because it depends on Agricultural Household indicator
    .drop(['Crop Farming and Gardening expenses'], axis=1, inplace=False)
    # use 8 output variables and 24 input variables
    .drop(['Total Rice Expenditure', 'Total Fish and  marine products Expenditure', 'Fruit Expenditure', 'Restaurant and hotels Expenditure',
           'Alcoholic Beverages Expenditure', 'Tobacco Expenditure', 'Clothing, Footwear and Other Wear Expenditure',
           'Imputed House Rental Value', 'Transportation Expenditure', 'Miscellaneous Goods and Services Expenditure',
           'Special Occasions Expenditure'], axis=1, inplace=False)
    # add input and output labels to each column
    .pipe(add_input_output_family_income)
    # normalize all the columns
    .pipe(normalize_columns_income)
    # remove all datapoints that do not satisfy the constraints
    .pipe(check_constraints_income)
    # sample 2500 examples
    .pipe(sample_2500_examples)
    # transpose such that the multiindex is indeed an index and not a column
    .transpose()
)

family_income_and_expenditure_reduced.to_csv('./Sample Family Income and Expenditure Reduced.csv')


