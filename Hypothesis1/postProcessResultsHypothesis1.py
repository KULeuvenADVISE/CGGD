"""
Script that computes the mean and standard deviation of the obtained results.

:author: Quinten Van Baelen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import numpy as np
import pandas as pd
import pickle

number_weight_initializations = 4
number_dataset_realizations = 1


def convert_mean_std_frame_to_latex(df, vertical_lines=False):
    temp_df = df.copy()
    temp_df = temp_df.transpose().droplevel(level=1)
    temp_df = temp_df.transpose()
    temp_df = temp_df.transpose().groupby(level=0).first().transpose()
    n = len(temp_df.columns)
    cols = 'l' + 'c' * n
    if vertical_lines:
        temp_latex_code = temp_df.to_latex(escape=False, column_format='|' + '|'.join(cols) + '|')
    else:
        temp_latex_code = temp_df.to_latex(escape=False, column_format=cols)
    return temp_latex_code


def combine_mean_std(df):
    temp_df = df.copy()
    temp_df['CGGD'] = temp_df['CGGD'].apply(lambda x: ' $\pm$ '.join(x.astype(str)), axis=1)
    temp_df['Baseline'] = temp_df['Baseline'].apply(lambda x: ' $\pm$ '.join(x.astype(str)), axis=1)
    temp_df['Fuzzy'] = temp_df['Fuzzy'].apply(lambda x: ' $\pm$ '.join(x.astype(str)), axis=1)
    return temp_df


def construct_results(name_experiment, name_dataset, train_sizes, train_sizes_full, directory_experiment='./', number_weight_initializations=1,
                      number_of_dataset_realisations=1, loss_boolean=True, satisfaction_ratio_boolean=True, print_console_boolean=False,
                      vertical_lines_table=False):
    """Construct LaTeX code for the mean and standard deviation of the results."""
    if loss_boolean:
        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'

        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + train_sizes[i] + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        temp_results.append(loaded_results_con['loss_value_test'])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_con = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_con = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Unconstrained/History'

        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + train_sizes[i] + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        temp_results.append(loaded_results_con['loss_value_test'])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_uncon = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_uncon = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Fuzzy/History'

        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + train_sizes[i] + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        temp_results.append(loaded_results_con['loss_value_test'])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_fuzzy = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_fuzzy = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        numpy_array_results = np.hstack([mean_con, std_con, mean_uncon, std_uncon, mean_fuzzy, std_fuzzy])

        array_results_df = pd.DataFrame(data=numpy_array_results, index=[str(size) for size in train_sizes_full], columns=['mean', 'std', 'mean', 'std', 'mean', 'std'])
        list_top_index = ['CGGD'] * 2 + ['Baseline'] * 2 + ['Fuzzy'] * 2
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std', 'mean', 'std', 'mean', 'std']])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes_full])
        mean_std_df_results_loss = mean_std_df.copy()

        latex_code_loss = (
            mean_std_df
                .transpose()
                .pipe(combine_mean_std)
                .pipe(convert_mean_std_frame_to_latex, vertical_lines=vertical_lines_table)
        )

        if print_console_boolean:
            print(latex_code_loss)
    else:
        latex_code_loss = ' '
        mean_std_df_results_loss = np.zeros((len(train_sizes), 6))

    if satisfaction_ratio_boolean:
        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'

        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + train_sizes[i] + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        temp_results.append(loaded_results_con['sat_ratio_test'])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_con = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_con = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Unconstrained/History'

        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + train_sizes[i] + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        temp_results.append(loaded_results_con['sat_ratio_test'])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_uncon = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_uncon = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Fuzzy/History'

        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + train_sizes[i] + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        temp_results.append(loaded_results_con['sat_ratio_test'])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_fuzzy = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_fuzzy = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        numpy_array_results = np.hstack([mean_con, std_con, mean_uncon, std_uncon, mean_fuzzy, std_fuzzy])

        array_results_df = pd.DataFrame(data=numpy_array_results, index=[str(size) for size in train_sizes_full], columns=['mean', 'std', 'mean', 'std', 'mean', 'std'])
        list_top_index = ['CGGD'] * 2 + ['Baseline'] * 2 + ['Fuzzy'] * 2
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std', 'mean', 'std', 'mean', 'std']])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes_full])
        mean_std_df_results_sr = mean_std_df.copy()

        latex_code_satisfaction = (
            mean_std_df
            .transpose()
            .pipe(combine_mean_std)
            .pipe(convert_mean_std_frame_to_latex, vertical_lines=vertical_lines_table)
        )

        if print_console_boolean:
            print(latex_code_satisfaction)
    else:
        latex_code_satisfaction = ''
        mean_std_df_results_sr = np.zeros((len(train_sizes), 6))

    return latex_code_loss, latex_code_satisfaction, mean_std_df_results_loss, mean_std_df_results_sr


print('-------------BIAS-CORRECTION--------------')
used_train_sizes = [0.1]
train_sizes_full = [200]
exp_latex_code_loss_bias, exp_latex_code_satisfaction_bias, results_loss_bias, results_sr_bias = construct_results(name_experiment='Hypothesis1_BiasCorrection',
                                                                               name_dataset='Sample_Bias_correction_ucl_reduced',
                                                                               train_sizes=[str(sizes) for sizes in used_train_sizes],
                                                                               train_sizes_full=[str(sizes) for sizes in train_sizes_full],
                                                                               directory_experiment='../Results',
                                                                               number_weight_initializations=4,
                                                                               number_of_dataset_realisations=1,
                                                                               loss_boolean=True,
                                                                               satisfaction_ratio_boolean=True,
                                                                               print_console_boolean=True,
                                                                               vertical_lines_table=False)

print('-------------FAMILY-INCOME-----------------')
used_train_sizes = [0.1]
train_sizes_full = [200]
exp_latex_code_loss_income, exp_latex_code_satisfaction_income, results_loss_income, results_sr_income = construct_results(name_experiment='Hypothesis1_FamilyIncome',
                                                                                   name_dataset='Sample Family Income and Expenditure Reduced',
                                                                                   train_sizes=[str(sizes) for sizes in used_train_sizes],
                                                                                   train_sizes_full=[str(sizes) for sizes in train_sizes_full],
                                                                                   directory_experiment='../Results',
                                                                                   number_weight_initializations=4,
                                                                                   number_of_dataset_realisations=1,
                                                                                   loss_boolean=True,
                                                                                   satisfaction_ratio_boolean=True,
                                                                                   print_console_boolean=True,
                                                                                   vertical_lines_table=False)


