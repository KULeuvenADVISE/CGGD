"""

:author: Quinten Van Baelen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import os
import tensorflow as tf
import aux as aux

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Poroelastic materials dataset
directory_dataset = '../DataSets/'
name_dataset = 'Sample Family Income and Expenditure Reduced'  # update this when files are given better names
extension_dataset = '.csv'

# define pseudo random seed for running the experiment
random_seed = 5

# define the margin for the bound constraints
margin = 0.1

# define hyperparameters
initial_learning_rate = 0.001
test_size = 0.1
validation_size = 0.1
max_num_epochs = 350
max_epoch_constraints = 500
batch_size = 100
acceptable_sat_ratio = 1
max_bumps_con = 3
max_time_after_bumps_con = 50
max_constant_perf_con = 150
max_restarts_con = 5
number_dataset_realizations = 1  # number of different train, validation and test sets considered
number_weight_initializations = 4  # number of different networks considered for each realization of the train, validation and test set
train_size = [1, 0.75, 0.5, 0.25, 0.1]
name_experiment = 'Hypothesis1_FamilyIncome'

constrained_dataset = True      # set True to run the constrained model for dataset
unconstrained_dataset = True    # set True to run the unconstrained model for dataset, it is required to have run the constrained model
                                # (so with the same initialization first because the number of epochs for the unconstrained training is
                                # equal to the number of epochs for which the corresponding constrained model was trained
fuzzy_dataset = True            # set True to run the fuzzy model for dataset

# Perform experiment on dataset
print('================================================================================================================')
print('Experiment on dataset (called ' + name_dataset + ') started.')

if constrained_dataset:
    print('----------------------------------------------------------------------------------------------------------------')
    print('Constrained case:')
    print('-----------------')
    aux.perform_experiment_different_training_sizes(directory_dataset=directory_dataset, name_dataset=name_dataset,
                                                    extension_dataset=extension_dataset, initial_learning_rate=initial_learning_rate,
                                                    test_size=test_size, validation_size=validation_size, max_num_epochs=max_num_epochs,
                                                    max_epoch_constraints=max_epoch_constraints, batch_size=batch_size,
                                                    acceptable_sat_ratio=acceptable_sat_ratio, max_bumps_con=max_bumps_con,
                                                    max_time_after_bumps_con=max_time_after_bumps_con, max_constant_perf_con=max_constant_perf_con,
                                                    max_restarts_con=max_restarts_con, random_state=random_seed,
                                                    number_dataset_realizations=number_dataset_realizations,
                                                    number_weight_initializations=number_weight_initializations, name_experiment=name_experiment,
                                                    constrained=constrained_dataset, unconstrained=False, fuzzy=False, margin=margin,
                                                    train_size=train_size)

if unconstrained_dataset:
    print('----------------------------------------------------------------------------------------------------------------')
    print('Unconstrained case:')
    aux.perform_experiment_different_training_sizes(directory_dataset=directory_dataset, name_dataset=name_dataset,
                                                    extension_dataset=extension_dataset, initial_learning_rate=initial_learning_rate,
                                                    test_size=test_size, validation_size=validation_size, max_num_epochs=max_num_epochs,
                                                    max_epoch_constraints=max_epoch_constraints, batch_size=batch_size,
                                                    acceptable_sat_ratio=acceptable_sat_ratio, max_bumps_con=max_bumps_con,
                                                    max_time_after_bumps_con=max_time_after_bumps_con, max_constant_perf_con=max_constant_perf_con,
                                                    max_restarts_con=max_restarts_con, random_state=random_seed,
                                                    number_dataset_realizations=number_dataset_realizations,
                                                    number_weight_initializations=number_weight_initializations,
                                                    name_experiment=name_experiment, constrained=False, unconstrained=unconstrained_dataset,
                                                    fuzzy=False, margin=margin, train_size=train_size)

if fuzzy_dataset:
    print('----------------------------------------------------------------------------------------------------------------')
    print('Fuzzy case:')
    aux.perform_experiment_different_training_sizes(directory_dataset=directory_dataset, name_dataset=name_dataset,
                                                    extension_dataset=extension_dataset, initial_learning_rate=initial_learning_rate,
                                                    test_size=test_size, validation_size=validation_size, max_num_epochs=max_num_epochs,
                                                    max_epoch_constraints=max_epoch_constraints, batch_size=batch_size,
                                                    acceptable_sat_ratio=acceptable_sat_ratio, max_bumps_con=max_bumps_con,
                                                    max_time_after_bumps_con=max_time_after_bumps_con, max_constant_perf_con=max_constant_perf_con,
                                                    max_restarts_con=max_restarts_con, random_state=random_seed,
                                                    number_dataset_realizations=number_dataset_realizations,
                                                    number_weight_initializations=number_weight_initializations,
                                                    name_experiment=name_experiment, constrained=False, unconstrained=False, fuzzy=fuzzy_dataset,
                                                    margin=margin, train_size=train_size)

print('================================================================================================================')
print('Experiment has ended.')
print('================================================================================================================')


