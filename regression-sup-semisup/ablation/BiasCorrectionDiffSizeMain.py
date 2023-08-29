"""Copyright (c) DTAI - KU Leuven - All right reserved.

Proprietary, do not copy or distribute without permission.

Written by Quinten Van Baelen ORCID iD 0000-0003-2863-4227, 2022."""

import os
import BiasCorrectionAux as aux


def init_keras(devices="", v=2):
    if v == 2: #tf2
        import tensorflow as tf
        from tensorflow.keras import backend as K
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    else: #tf1
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        from tensorflow.keras import backend as K
        import tensorflow as tf
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
    return K

# GPU config
gpu_on = True
if gpu_on:
    init_keras(devices="0")  # choose a different integer if you have multiple GPUs and want to use a different one
else:
    init_keras(devices="-1")

# For each dataset that needs to be ran, define the directory, name and extension as shown below in order to have the results be formatted
# in separate files.

# dataset
directory_dataset = '../../DataSets/'
name_dataset = 'Sample_Bias_correction_ucl_reduced'  # update this when files are given better names
extension_dataset = '.csv'

# define pseudo random seed for running the experiment
random_seed = 15

# define the margin for the bound constraints
margin = 0.1

# define hyperparameters
initial_learning_rate = 0.001
test_size = 0.1
validation_size = 0.1
max_num_epochs = 1250
max_epoch_constraints = 1000
batch_size = 100
acceptable_sat_ratio = 0.7
max_bumps_con = 8
max_time_after_bumps_con = 100
max_constant_perf_con = 250
max_restarts_con = 10
number_dataset_realizations = 1
number_weight_initializations = 4
train_size = [1, 0.75, 0.5, 0.25, 0.1]
rescale_factors = [1.2, 1.5, 5, 20, 500]
name_experiment = 'BiasCorrection_DiffSize_DiffRescaleFactors'

constrained_dataset = True

# Perform experiment on dataset
print('================================================================================================================')
print('Experiment on dataset (called ' + name_dataset + ') started.')

if constrained_dataset:
    print('----------------------------------------------------------------------------------------------------------------')
    print('Constrained case:')
    print('-----------------')
    aux.perform_experiment_different_training_sizes_different_rescale_factors(directory_dataset, name_dataset, extension_dataset,
                                                                              initial_learning_rate, test_size, validation_size, train_size,
                                                                              max_num_epochs, max_epoch_constraints, batch_size, acceptable_sat_ratio,
                                                                              max_bumps_con, max_time_after_bumps_con, max_constant_perf_con,
                                                                              max_restarts_con, random_state=random_seed,
                                                                              number_dataset_realizations=number_dataset_realizations,
                                                                              number_weight_initializations=number_weight_initializations,
                                                                              name_experiment=name_experiment, margin=margin,
                                                                              rescale_factors=rescale_factors)

print('================================================================================================================')
print('Experiment has ended.')
print('================================================================================================================')
