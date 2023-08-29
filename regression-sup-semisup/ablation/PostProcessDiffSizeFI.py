"""Copyright (c) DTAI - KU Leuven - All right reserved.

Proprietary, do not copy or distribute without permission.

Written by Quinten Van Baelen ORCID iD 0000-0003-2863-4227, 2022."""

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
    temp_df['Unconstrained model'] = temp_df['Unconstrained model'].apply(lambda x: ' $\pm$ '.join(x.astype(str)), axis=1)
    temp_df['Fuzzy'] = temp_df['Fuzzy'].apply(lambda x: ' $\pm$ '.join(x.astype(str)), axis=1)
    return temp_df


def combine_mean_std_rescale_factors(df):
    rescale_factors = [1.2, 1.5, 5, 20, 500]
    temp_df = df.copy()
    for r in rescale_factors:
        temp_df[str(r)] = temp_df[str(r)].apply(lambda x: ' $\pm$ '.join(x.astype(str)), axis=1)
    return temp_df


def construct_results_test(name_experiment, name_dataset, train_sizes, directory_experiment='./', number_weight_initializations=1,
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
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

        array_results_df = pd.DataFrame(data=numpy_array_results, index=[str(size) for size in train_sizes], columns=['mean', 'std', 'mean', 'std', 'mean', 'std'])
        list_top_index = ['CGGD'] * 2 + ['Unconstrained model'] * 2 + ['Fuzzy'] * 2
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std', 'mean', 'std', 'mean', 'std']])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
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

        array_results_df = pd.DataFrame(data=numpy_array_results, index=[str(size) for size in train_sizes], columns=['mean', 'std', 'mean', 'std', 'mean', 'std'])
        list_top_index = ['CGGD'] * 2 + ['Unconstrained model'] * 2 + ['Fuzzy'] * 2
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std', 'mean', 'std', 'mean', 'std']])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
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


def construct_results_test_rescale_factors(name_experiment, name_dataset, train_sizes, rescale_factors, directory_experiment='./',
                                           number_weight_initializations=1, number_of_dataset_realisations=1, loss_boolean=True,
                                           satisfaction_ratio_boolean=True, print_console_boolean=False, vertical_lines_table=False):
    """Construct LaTeX code for the mean and standard deviation of the results."""
    if loss_boolean:
        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'

        r = rescale_factors[0]
        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle',
                              'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['loss_value_test'])
            all_results.append(temp_results)

        del r

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        np_results = np.hstack([mean_np, std_np])

        for r in rescale_factors[1:]:
            all_results = []
            for i in range(0, len(train_sizes)):
                temp_results = []
                for j in range(0, number_weight_initializations):
                    for k in range(0, number_of_dataset_realisations):
                        with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle', 'rb') as handle:
                            loaded_results_con = pickle.load(handle)
                            best_epoch = loaded_results_con['best_epoch']
                            temp_results.append(loaded_results_con['loss_value_test'])
                all_results.append(temp_results)

            all_results_df = pd.DataFrame(np.asarray(all_results))
            all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
            mean_sr = all_results_df.mean()
            std_sr = all_results_df.std()

            temp_mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
            temp_std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

            np_results = np.hstack([np_results, temp_mean_np, temp_std_np])

        array_results_df = pd.DataFrame(data=np_results, index=[str(size) for size in train_sizes], columns=['mean', 'std'] * len(rescale_factors))
        list_top_index = [str(r) for r in rescale_factors for _ in (0, 1)]
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std'] * len(rescale_factors)])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
        mean_std_df_results_loss = mean_std_df.copy()

        latex_code_loss = (
            mean_std_df
                .transpose()
                .pipe(combine_mean_std_rescale_factors)
                .pipe(convert_mean_std_frame_to_latex, vertical_lines=vertical_lines_table)
        )

        if print_console_boolean:
            print(latex_code_loss)
    else:
        latex_code_loss = ' '
        mean_std_df_results_loss = np.zeros((len(train_sizes), 10))

    if satisfaction_ratio_boolean:
        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'

        r = rescale_factors[0]
        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['sat_ratio_test'])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        np_results = np.hstack([mean_np, std_np])

        del r

        for r in rescale_factors[1:]:
            all_results = []
            for i in range(0, len(train_sizes)):
                temp_results = []
                for j in range(0, number_weight_initializations):
                    for k in range(0, number_of_dataset_realisations):
                        with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle', 'rb') as handle:
                            loaded_results_con = pickle.load(handle)
                            best_epoch = loaded_results_con['best_epoch']
                            temp_results.append(loaded_results_con['sat_ratio_test'])
                all_results.append(temp_results)

            all_results_df = pd.DataFrame(np.asarray(all_results))
            all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
            mean_sr = all_results_df.mean()
            std_sr = all_results_df.std()

            temp_mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
            temp_std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

            np_results = np.hstack([np_results, temp_mean_np, temp_std_np])

        array_results_df = pd.DataFrame(data=np_results, index=[str(size) for size in train_sizes], columns=['mean', 'std'] * len(rescale_factors))
        list_top_index = [str(r) for r in rescale_factors for _ in (0, 1)]
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std'] * len(rescale_factors)])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
        mean_std_df_results_sr = mean_std_df.copy()

        latex_code_satisfaction = (
            mean_std_df
            .transpose()
            .pipe(combine_mean_std_rescale_factors)
            .pipe(convert_mean_std_frame_to_latex, vertical_lines=vertical_lines_table)
        )

        if print_console_boolean:
            print(latex_code_satisfaction)
    else:
        latex_code_satisfaction = ''
        mean_std_df_results_sr = np.zeros((len(train_sizes), 10))

    return latex_code_loss, latex_code_satisfaction, mean_std_df_results_loss, mean_std_df_results_sr


def construct_results_train(name_experiment, name_dataset, train_sizes, directory_experiment='./', number_weight_initializations=1,
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['train_loss_results'][best_epoch-1])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['train_loss_results'][best_epoch - 1])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['train_loss_results'][best_epoch - 1])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_fuzzy = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_fuzzy = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        numpy_array_results = np.hstack([mean_con, std_con, mean_uncon, std_uncon, mean_fuzzy, std_fuzzy])

        array_results_df = pd.DataFrame(data=numpy_array_results, index=[str(size) for size in train_sizes], columns=['mean', 'std', 'mean', 'std', 'mean', 'std'])
        list_top_index = ['CGGD'] * 2 + ['Unconstrained model'] * 2 + ['Fuzzy'] * 2
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std', 'mean', 'std', 'mean', 'std']])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['train_satisfaction_results'][best_epoch - 1])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['train_satisfaction_results'][best_epoch - 1])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['train_satisfaction_results'][best_epoch - 1])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_fuzzy = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_fuzzy = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        numpy_array_results = np.hstack([mean_con, std_con, mean_uncon, std_uncon, mean_fuzzy, std_fuzzy])

        array_results_df = pd.DataFrame(data=numpy_array_results, index=[str(size) for size in train_sizes], columns=['mean', 'std', 'mean', 'std', 'mean', 'std'])
        list_top_index = ['CGGD'] * 2 + ['Unconstrained model'] * 2 + ['Fuzzy'] * 2
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std', 'mean', 'std', 'mean', 'std']])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
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


def construct_results_train_rescale_factors(name_experiment, name_dataset, train_sizes, rescale_factors, directory_experiment='./',
                                            number_weight_initializations=1, number_of_dataset_realisations=1, loss_boolean=True,
                                            satisfaction_ratio_boolean=True, print_console_boolean=False, vertical_lines_table=False):
    """Construct LaTeX code for the mean and standard deviation of the results."""
    if loss_boolean:
        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'

        r = rescale_factors[0]
        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle',
                              'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        # temp_results.append(list(map(np.float32, loaded_results_con['loss_value_test'])))
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['train_loss_results'][best_epoch - 1])
            all_results.append(temp_results)

        del r

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        np_results = np.hstack([mean_np, std_np])

        for r in rescale_factors[1:]:
            all_results = []
            for i in range(0, len(train_sizes)):
                temp_results = []
                for j in range(0, number_weight_initializations):
                    for k in range(0, number_of_dataset_realisations):
                        with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle', 'rb') as handle:
                            loaded_results_con = pickle.load(handle)
                            best_epoch = loaded_results_con['best_epoch']
                            temp_results.append(loaded_results_con['train_loss_results'][best_epoch-1])
                all_results.append(temp_results)

            all_results_df = pd.DataFrame(np.asarray(all_results))
            all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
            mean_sr = all_results_df.mean()
            std_sr = all_results_df.std()

            temp_mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
            temp_std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

            np_results = np.hstack([np_results, temp_mean_np, temp_std_np])

        array_results_df = pd.DataFrame(data=np_results, index=[str(size) for size in train_sizes], columns=['mean', 'std'] * len(rescale_factors))
        list_top_index = [str(r) for r in rescale_factors for _ in (0, 1)]
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std'] * len(rescale_factors)])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
        mean_std_df_results_loss = mean_std_df.copy()

        latex_code_loss = (
            mean_std_df
            .transpose()
            .pipe(combine_mean_std_rescale_factors)
            .pipe(convert_mean_std_frame_to_latex, vertical_lines=vertical_lines_table)
        )

        if print_console_boolean:
            print(latex_code_loss)
    else:
        latex_code_loss = ' '
        mean_std_df_results_loss = np.zeros((len(train_sizes), 10))

    if satisfaction_ratio_boolean:
        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'

        r = rescale_factors[0]
        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['train_satisfaction_results'][best_epoch - 1])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        np_results = np.hstack([mean_np, std_np])

        del r

        for r in rescale_factors[1:]:
            all_results = []
            for i in range(0, len(train_sizes)):
                temp_results = []
                for j in range(0, number_weight_initializations):
                    for k in range(0, number_of_dataset_realisations):
                        with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle', 'rb') as handle:
                            loaded_results_con = pickle.load(handle)
                            best_epoch = loaded_results_con['best_epoch']
                            temp_results.append(loaded_results_con['train_satisfaction_results'][best_epoch - 1])
                all_results.append(temp_results)

            all_results_df = pd.DataFrame(np.asarray(all_results))
            all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
            mean_sr = all_results_df.mean()
            std_sr = all_results_df.std()

            temp_mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
            temp_std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

            np_results = np.hstack([np_results, temp_mean_np, temp_std_np])

        array_results_df = pd.DataFrame(data=np_results, index=[str(size) for size in train_sizes], columns=['mean', 'std'] * len(rescale_factors))
        list_top_index = [str(r) for r in rescale_factors for _ in (0, 1)]
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std'] * len(rescale_factors)])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
        mean_std_df_results_sr = mean_std_df.copy()

        latex_code_satisfaction = (
            mean_std_df
            .transpose()
            .pipe(combine_mean_std_rescale_factors)
            .pipe(convert_mean_std_frame_to_latex, vertical_lines=vertical_lines_table)
        )

        if print_console_boolean:
            print(latex_code_satisfaction)
    else:
        latex_code_satisfaction = ''
        mean_std_df_results_sr = np.zeros((len(train_sizes), 8))

    return latex_code_loss, latex_code_satisfaction, mean_std_df_results_loss, mean_std_df_results_sr


def construct_results_val(name_experiment, name_dataset, train_sizes, directory_experiment='./', number_weight_initializations=1,
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['val_loss_results'][best_epoch-1])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['val_loss_results'][best_epoch - 1])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['val_loss_results'][best_epoch - 1])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_fuzzy = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_fuzzy = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        numpy_array_results = np.hstack([mean_con, std_con, mean_uncon, std_uncon, mean_fuzzy, std_fuzzy])

        array_results_df = pd.DataFrame(data=numpy_array_results, index=[str(size) for size in train_sizes], columns=['mean', 'std', 'mean', 'std', 'mean', 'std'])
        list_top_index = ['CGGD'] * 2 + ['Unconstrained model'] * 2 + ['Fuzzy'] * 2
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std', 'mean', 'std', 'mean', 'std']])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['val_satisfaction_results'][best_epoch - 1])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['val_satisfaction_results'][best_epoch - 1])
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
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['val_satisfaction_results'][best_epoch - 1])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_fuzzy = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_fuzzy = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        numpy_array_results = np.hstack([mean_con, std_con, mean_uncon, std_uncon, mean_fuzzy, std_fuzzy])

        array_results_df = pd.DataFrame(data=numpy_array_results, index=[str(size) for size in train_sizes], columns=['mean', 'std', 'mean', 'std', 'mean', 'std'])
        list_top_index = ['CGGD'] * 2 + ['Unconstrained model'] * 2 + ['Fuzzy'] * 2
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std', 'mean', 'std', 'mean', 'std']])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
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


def construct_results_val_rescale_factors(name_experiment, name_dataset, train_sizes, rescale_factors, directory_experiment='./',
                                          number_weight_initializations=1, number_of_dataset_realisations=1, loss_boolean=True,
                                          satisfaction_ratio_boolean=True, print_console_boolean=False, vertical_lines_table=False):
    """Construct LaTeX code for the mean and standard deviation of the results."""
    if loss_boolean:
        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'

        r = rescale_factors[0]
        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle',
                              'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['val_loss_results'][best_epoch - 1])
            all_results.append(temp_results)

        del r

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        np_results = np.hstack([mean_np, std_np])

        for r in rescale_factors[1:]:
            all_results = []
            for i in range(0, len(train_sizes)):
                temp_results = []
                for j in range(0, number_weight_initializations):
                    for k in range(0, number_of_dataset_realisations):
                        with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle', 'rb') as handle:
                            loaded_results_con = pickle.load(handle)
                            best_epoch = loaded_results_con['best_epoch']
                            temp_results.append(loaded_results_con['val_loss_results'][best_epoch-1])
                all_results.append(temp_results)

            all_results_df = pd.DataFrame(np.asarray(all_results))
            all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
            mean_sr = all_results_df.mean()
            std_sr = all_results_df.std()

            temp_mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
            temp_std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

            np_results = np.hstack([np_results, temp_mean_np, temp_std_np])

        array_results_df = pd.DataFrame(data=np_results, index=[str(size) for size in train_sizes], columns=['mean', 'std'] * len(rescale_factors))
        list_top_index = [str(r) for r in rescale_factors for _ in (0, 1)]
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std'] * len(rescale_factors)])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
        mean_std_df_results_loss = mean_std_df.copy()

        latex_code_loss = (
            mean_std_df
                .transpose()
                .pipe(combine_mean_std_rescale_factors)
                .pipe(convert_mean_std_frame_to_latex, vertical_lines=vertical_lines_table)
        )

        if print_console_boolean:
            print(latex_code_loss)
    else:
        latex_code_loss = ' '
        mean_std_df_results_loss = np.zeros((len(train_sizes), 8))

    if satisfaction_ratio_boolean:
        all_results = []

        file = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'

        r = rescale_factors[0]
        for i in range(0, len(train_sizes)):
            temp_results = []
            for j in range(0, number_weight_initializations):
                for k in range(0, number_of_dataset_realisations):
                    with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle', 'rb') as handle:
                        loaded_results_con = pickle.load(handle)
                        best_epoch = loaded_results_con['best_epoch']
                        temp_results.append(loaded_results_con['val_satisfaction_results'][best_epoch - 1])
            all_results.append(temp_results)

        all_results_df = pd.DataFrame(np.asarray(all_results))
        all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
        mean_sr = all_results_df.mean()
        std_sr = all_results_df.std()

        mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
        std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

        np_results = np.hstack([mean_np, std_np])

        del r

        for r in rescale_factors[1:]:
            all_results = []
            for i in range(0, len(train_sizes)):
                temp_results = []
                for j in range(0, number_weight_initializations):
                    for k in range(0, number_of_dataset_realisations):
                        with open(file + '_' + str(j) + '_dataset_' + str(k) + '_size_' + str(train_sizes[i]) + '_rescale_factor_' + str(r) + '.pickle', 'rb') as handle:
                            loaded_results_con = pickle.load(handle)
                            best_epoch = loaded_results_con['best_epoch']
                            temp_results.append(loaded_results_con['val_satisfaction_results'][best_epoch - 1])
                all_results.append(temp_results)

            all_results_df = pd.DataFrame(np.asarray(all_results))
            all_results_df = all_results_df.transpose()  # have the columns correspond to the results of the different models for a dataset
            mean_sr = all_results_df.mean()
            std_sr = all_results_df.std()

            temp_mean_np = np.reshape(mean_sr.to_numpy(), (len(train_sizes), 1))
            temp_std_np = np.reshape(std_sr.to_numpy(), (len(train_sizes), 1))

            np_results = np.hstack([np_results, temp_mean_np, temp_std_np])

        array_results_df = pd.DataFrame(data=np_results, index=[str(size) for size in train_sizes], columns=['mean', 'std'] * len(rescale_factors))
        list_top_index = [str(r) for r in rescale_factors for _ in (0, 1)]
        multi_index = pd.MultiIndex.from_arrays([list_top_index, ['mean', 'std'] * len(rescale_factors)])
        mean_std_df = pd.DataFrame(array_results_df.transpose().to_numpy(), index=multi_index, columns=[str(size) for size in train_sizes])
        mean_std_df_results_sr = mean_std_df.copy()

        latex_code_satisfaction = (
            mean_std_df
            .transpose()
            .pipe(combine_mean_std_rescale_factors)
            .pipe(convert_mean_std_frame_to_latex, vertical_lines=vertical_lines_table)
        )

        if print_console_boolean:
            print(latex_code_satisfaction)
    else:
        latex_code_satisfaction = ''
        mean_std_df_results_sr = np.zeros((len(train_sizes), 8))

    return latex_code_loss, latex_code_satisfaction, mean_std_df_results_loss, mean_std_df_results_sr


print('-------------FAMILY-INCOME-TRAIN------------')
used_train_sizes = [1, 0.75, 0.5, 0.25, 0.1]
rescale_factors = [1.2, 1.5, 5, 20, 500]
exp_latex_code_loss_income_train, exp_latex_code_satisfaction_income_train, results_loss_income_train, results_sr_income_train = \
    construct_results_train_rescale_factors(name_experiment='FamilyIncome_DiffSize_DiffRescaleFactors',
                                            name_dataset='Sample Family Income and Expenditure Reduced',
                                            train_sizes=[str(sizes) for sizes in used_train_sizes],
                                            rescale_factors=rescale_factors,
                                            directory_experiment='./Results',
                                            number_weight_initializations=4,
                                            number_of_dataset_realisations=1,
                                            loss_boolean=True,
                                            satisfaction_ratio_boolean=True,
                                            print_console_boolean=True,
                                            vertical_lines_table=False)

print('-------------FAMILY-INCOME-VALIDATION------------')
used_train_sizes = [1, 0.75, 0.5, 0.25, 0.1]
rescale_factors = [1.2, 1.5, 5, 20, 500]
exp_latex_code_loss_income_val, exp_latex_code_satisfaction_income_val, results_loss_income_val, results_sr_income_val = \
    construct_results_val_rescale_factors(name_experiment='FamilyIncome_DiffSize_DiffRescaleFactors',
                                          name_dataset='Sample Family Income and Expenditure Reduced',
                                          train_sizes=[str(sizes) for sizes in used_train_sizes],
                                          rescale_factors=rescale_factors,
                                          directory_experiment='./Results',
                                          number_weight_initializations=4,
                                          number_of_dataset_realisations=1,
                                          loss_boolean=True,
                                          satisfaction_ratio_boolean=True,
                                          print_console_boolean=True,
                                          vertical_lines_table=False)

print('-------------FAMILY-INCOME-TEST------------')
used_train_sizes = [1, 0.75, 0.5, 0.25, 0.1]
rescale_factors = [1.2, 1.5, 5, 20, 500]
exp_latex_code_loss_income_test, exp_latex_code_satisfaction_income_test, results_loss_income_test, results_sr_income_test = \
    construct_results_test_rescale_factors(name_experiment='FamilyIncome_DiffSize_DiffRescaleFactors',
                                           name_dataset='Sample Family Income and Expenditure Reduced',
                                           train_sizes=[str(sizes) for sizes in used_train_sizes],
                                           rescale_factors=rescale_factors,
                                           directory_experiment='./Results',
                                           number_weight_initializations=4,
                                           number_of_dataset_realisations=1,
                                           loss_boolean=True,
                                           satisfaction_ratio_boolean=True,
                                           print_console_boolean=True,
                                           vertical_lines_table=False)



