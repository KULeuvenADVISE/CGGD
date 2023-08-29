"""Copyright (c) DTAI - KU Leuven - All right reserved.

Proprietary, do not copy or distribute without permission.

Written by Quinten Van Baelen ORCID iD 0000-0003-2863-4227, 2021."""

import numpy as np
import os
import scipy.io as io
import pickle
import tensorflow as tf  # requires tf >=2.2
import pandas as pd
import BiasCorrectionCGGD as cggd
import BiasCorrectionConstraints as bcc
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import he_normal, he_uniform
from random import seed, randint, random, sample
import math


def prep_data(prep_directory, prep_name, prep_ext):
    """Load in the data.

    Parameters
    ----------
    prep_directory : string
        String denoting the directory in which the dataset is saved.
    prep_name : string
        String denoting the name of the file in which the dataset is saved.
    prep_ext : string
        String denoting the extension of the savefile. Supported extension are .mat and .csv.

    Returns
    ----------
    prep_input : np.ndarray
        Numpy array containing the input of all the instances.
    prep_output : np.ndarray
        Numpy array containing the output of all the instances.

    """
    if prep_ext == '.mat':
        prep_data_loaded = io.loadmat(prep_directory + prep_name)
        prep_input = prep_data_loaded.get('Imx')
        prep_output = prep_data_loaded.get('Omx')
    elif prep_ext == '.csv':
        prep_data_loaded = pd.read_csv(prep_directory + prep_name + '.csv', index_col=[0, 1], header=0)
        prep_data_loaded = prep_data_loaded.transpose()
        prep_input = prep_data_loaded['Input'].to_numpy()
        prep_output = prep_data_loaded['Output'].to_numpy()
    else:
        print('Extension is not supported.')
        prep_input = []
        prep_output = []

    return prep_input, prep_output


def split_train_test_val(split_input_data, split_output_data, split_test_size, split_val_size, split_random_state1, split_random_state2):
    """Split the dataset into train, validation and test sets.

    Parameters
    ----------
    split_input_data : np.ndarray
        Numpy array containing the input of all instances in the dataset.
    split_output_data : np.ndarray
        Numpy array containing the output of all instances in the dataset.
    split_test_size : float
        Float in (0,1) denoting the test size relative to the size of the total dataset.
    split_val_size : float
        Float in (0,1) denoting the validation size relative to the size of the total dataset.
    split_random_state1 : int
        Integer denoting the pseudo random seed used to determine the test set.
    split_random_state2 : int
        Integer denoting the pseudo random seed used to determine the train and validation set.

    Returns
    ----------
    split_train_input : np.ndarray
        Numpy array containing the input of all instances in the train set.
    split_val_input : np.ndarray
        Numpy array containing the input of all instances in the validation set.
    split_test_input : np.ndarray
        Numpy array containing the input of all instances in the test set.
    split_train_output : np.ndarray
        Numpy array containing the output of all instances in the train set.
    split_val_output : np.ndarray
        Numpy array containing the output of all instances in the validation set.
    split_test_output : np.ndarray
        Numpy array containing the output of all instances in the test set.

    """

    split_train_val_input, split_test_input, split_train_val_output, split_test_output = \
        train_test_split(split_input_data,
                         split_output_data,
                         test_size=split_test_size,
                         random_state=split_random_state1)
    split_train_input, split_val_input, split_train_output, split_val_output = \
        train_test_split(split_train_val_input,
                         split_train_val_output,
                         test_size=split_val_size/(1-split_test_size),
                         random_state=split_random_state2)
    return split_train_input, split_val_input, split_test_input, split_train_output, split_val_output, split_test_output


def split_train(x_train, y_train, previous_size_reductions, new_test_size, pseudo_seed):
    """Reduce the training set in size.

    Parameters
    ----------
    x_train : np.ndarray
        Numpy array containing the input of all instances in the dataset.
    y_train : np.ndarray
        Numpy array containing the output of all instances in the dataset.
    previous_size_reductions : list[float]
        List of floats in [0,1] denoting the size of the previous reductions.
    new_test_size : float
        Float in (0,1] denoting the size of the new training size relative to the original size of the training set.
    pseudo_seed : list[int]
        List of integers denoting the pseudo random seeds used to choose the reductions of the training set.

    Returns
    ----------
    new_x_train : np.ndarray
        Numpy array containing the input of the reduced training set.
    new_y_train : np.ndarray
        Numpy array containing the output of the reduced training set.

    """
    new_x_train = x_train
    new_y_train = y_train
    previous_size = 1
    for i in range(0, len(previous_size_reductions)):
        new_x_train, _, new_y_train, _ = train_test_split(new_x_train,
                                                          new_y_train,
                                                          test_size=1-previous_size_reductions[i]/previous_size,
                                                          random_state=pseudo_seed[i])
        previous_size = previous_size_reductions[i]

    new_x_train, _, new_y_train, _ = train_test_split(new_x_train, new_y_train, test_size=(1-new_test_size)*previous_size, )

    return new_x_train, new_y_train


def make_dataset(x_data, y_data, batch_size):
    """Make a tf.data.Dataset of a given batch size. This function is used to convert the ndarrays of train, validation and test sets to
    another dtype that can be used by TensorFlow.

    Parameters
    ----------
    x_data : np.ndarray
        Numpy array containing the input of the instances in the dataset.
    y_data : np.ndarray
        Numpy array containing the output of the instances in the dataset.
    batch_size : int
        Integer denoting the batch size used.

    Returns
    ----------
    dataset : tf.data.Dataset
        Converted the arrays x_data, y_data to the right dtype for using in the training procedure.
    """

    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(x_data, dtype=tf.float32), tf.cast(y_data, dtype=tf.float32)))
    dataset = dataset.batch(batch_size)
    return dataset


def build_model_on_seed(custom_seed, input_dim, output_dim):
    """Define the architecture of the model where the weight and bias initialization depend on a pseudo random seed.

    Parameters
    ----------
    custom_seed : int
        Integer denoting the pseudo random seed.
    input_dim : int
        Integer denoting the input dimension of the network.
    output_dim : int
        Integer denoting the output dimension of the network.

    Returns
    ----------
    model : tf.keras.Model
        Model initialized on the pseudo random seed.

    """
    # the kernel initializer is fixed to be he_normal for RELU layers and to he_uniform for the final linear layer. If one wants to adjust
    # the initializer, then one should replace he_normal and he_uniform in the function below.
    # the biases of the RELU layers are initialized to 0.01 and for the linear layer to 0
    inputs = keras.layers.Input(shape=(input_dim,))
    seed(custom_seed)
    ran_min = round(random() * 10)
    ran_max = round(random() * (10 ** 6))
    x = keras.layers.Dense(35,
                           activation='relu',
                           kernel_initializer=he_normal(seed=randint(ran_min, ran_max)),
                           bias_initializer=tf.keras.initializers.Constant(value=0.01))(inputs)
    for i in range(1, 3):
        x = keras.layers.Dense(35,
                               activation='relu',
                               kernel_initializer=he_normal(seed=randint(ran_min, ran_max)),
                               bias_initializer=tf.keras.initializers.Constant(value=0.01))(x)

    outputs = keras.layers.Dense(output_dim,
                                 activation='linear',
                                 kernel_initializer=he_uniform(seed=randint(ran_min, ran_max)),
                                 bias_initializer=tf.keras.initializers.Constant(value=0))(x)

    model = keras.Model(inputs, outputs)

    return model


def pick_supervised_batches(percentage_unsupervised, number_of_batches, number_of_training_points, batch_size, random_seed):
    """In this implementation we assumed that a batch is either completely supervised or completely unsupervised. Therefore it is assumed
    that the percentage used for being unsupervised can be reached by making a certain amount of batches unsupervised.

    Parameters
    ----------
    percentage_unsupervised : float
        Float between 0 and 1 denoting the percentage of the data used as unsupervised datapoints.
    number_of_batches : int
        Integer denoting the total number of batches.
    number_of_training_points : int
        Integer denoting the total number of training points.
    batch_size : int
        Integer dentoing the batch size.
    random_seed : int
        Integer denoting the pseudo-random seed to use for picking which batches are supervised.

    Returns
    ----------
    indices_supervised : list[Int]
        List of integers denoting the indices of the batches that are used in a supervised way, that is, using the corresponding labels.

    """
    batches_to_pick = int(round(number_of_training_points * (1-percentage_unsupervised) / batch_size))
    seed(random_seed)
    if isinstance(batches_to_pick, int):
        indices_supervised = sample(list(range(0, number_of_batches)), k=batches_to_pick)
        return indices_supervised
    else:
        indices_supervised = []
        print('Returned none.')
        return indices_supervised


def perform_experiment_different_training_sizes_different_rescale_factors(directory_dataset, name_dataset, extension_dataset, initial_learning_rate,
                                                                          test_size, validation_size, train_size, max_num_epochs,
                                                                          max_epoch_constraints, batch_size, acceptable_sat_ratio, max_bumps_con,
                                                                          max_time_after_bumps_con, max_constant_perf_con, max_restarts_con,
                                                                          random_state, number_dataset_realizations, number_weight_initializations,
                                                                          name_experiment, margin,
                                                                          rescale_factors):
    """ Perform the experiment for the dataset at directory_experiment + name_experiment + extension_experiment.

    Nothing is returned since all results are saved into files.

    The training sets are determined from large to small and each time a part is removed from the current training set. The validation and test
    sets are kept the same over all experiments.

    Parameters
    ----------
    directory_dataset : string
        String representing the directory where the dataset is stored.
    name_dataset : string
        String representing the name of the save file of the dataset.
    extension_dataset : string
        String representing the extension of the save file of the dataset.
    initial_learning_rate : float
        Float giving the learning rate used at the beginning of the training procedure.
    test_size : float
        Float in (0,1) indicating the size of the test set compared to the total data available.
    validation_size : float
        Float in (0,1) indicating the size of the validation set compared to the total data available.
    train_size : list[float]
        List of floats in (0,1) indicating the size of the training set compared to the original size of the training set determined by the
        validation size and test size.
    max_num_epochs : int
        Integer defining the maximum number of epochs that can be used to optimized the loss function only. The loss function is only
        optimized when all the examples in the train set satisfy the constraints. The training process is stopped when this number is
        reached.
    max_epoch_constraints : int
        Integer defining the maximum number of epochs that can be used to optimize the constraints. The training process is stopped when
        this number is reached.
    batch_size : int
        Integer defining the batch size used for training and validation.
    acceptable_sat_ratio : float
        Float between [0,1] indicating how large the satisfaction ratio needs to be in order to allow for training the loss function only.
        Setting this value to 1 will result in the procedure as described in the corresponding paper. Setting this value to 0 will result in
        training without constraints.
    max_bumps_con : int
        Integer denoting the maximal number of bumps that are allowed before reverting the model to a previous best found model.
    max_time_after_bumps_con : int
        Integer denoting the maximum number of epochs that can be used to obtain a better model after leaving the acceptable model space. If
        this number is reached, then the model is reverted to a previous best found model.
    max_constant_perf_con : int
        Integer denoting how long the algorithm can train without improving the performance. This is used as an early-stopping criterion.
    max_restarts_con : int
        Integer denoting the maximal number of reverting of the model that can be done before ending the training procedure.
    random_state : int
        Integer denoting the pseudo random seed used to initialize all the random values.
    number_dataset_realizations : int
        Integer denoting the amount of realizations need to be performed for the dataset.
    number_weight_initializations : int
        Integer denoting the amount of different network initializations need to be performed.
    name_experiment : string
        String containing the name of the experiment, which is used as a directory to store all the results.
    margin : tf.Tensor
        Constant tensor of shape=(,) and dtype=tf.float32 used to adjust the bound constraints extracted from the dataset.
    rescale_factors : list[float]
        List of floats (all strictly larger than 1) defining the different rescale factors that will be used during the experiments.

    Returns
    ----------

    """
    constrained = True
    seed(random_state)
    ran_min = round(random() * 10)  # take random lower bound on the pseudo random seeds used
    ran_max = round(random() * (10 ** 6))  # take random upper bound on the pseudo random seeds used

    weight_states = []  # list containing the pseudo random seeds used to initialize the network weights
    dataset_states1 = []  # list containing the pseudo random seeds used to divide the data in a test set and an other set
    dataset_states2 = []  # list containing the pseudo random seeds used to divide the other set into a train set and a validation set.
    dataset_states3 = []  # list containing the pseudo random seeds used to take a part of the training set.

    for i in range(0, number_weight_initializations):
        weight_states.append(randint(ran_min, ran_max))

    for i in range(0, number_dataset_realizations):
        dataset_states1.append(randint(ran_min, ran_max))
        dataset_states2.append(randint(ran_min, ran_max))

    train_size.sort(reverse=True)  # make sure that the largest element is first and the other elements are decreasing
    for i in range(0, len(train_size)):
        dataset_states3.append(randint(ran_min, ran_max))

    tf.random.set_seed(randint(ran_min, ran_max))

    input_data, output_data = prep_data(prep_directory=directory_dataset, prep_name=name_dataset, prep_ext=extension_dataset)

    directory_experiment = './Results'

    if not os.path.isdir(directory_experiment):
        os.makedirs(directory_experiment)

    if not os.path.isdir(directory_experiment + '/' + name_experiment):
        os.makedirs(directory_experiment + '/' + name_experiment)

    if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset):
        os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset)

    if constrained:  # Put all the constrained models in a directory and the unconstrained models in a separate one
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained')
        save_location_model = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Model'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Model'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Model')
        save_location_training = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History')
        save_location_lr = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Learning_rate'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Learning_rate'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Learning_rate')
        save_location_aux_variables = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Aux_variables'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Aux_variables'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Aux_variables')
        save_location_hyperparameters = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Hyperparameters'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Hyperparameters'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Hyperparameters')
        if not os.path.isdir(save_location_hyperparameters + '/DifferentRescaleFactors/'):
            os.makedirs(save_location_hyperparameters + '/DifferentRescaleFactors/')
        save_location_bound_constraints = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Bound_constraints'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Bound_constraints'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Bound_constraints')
        if not os.path.isdir(save_location_bound_constraints + '/DifferentRescaleFactors/'):
            os.makedirs(save_location_bound_constraints + '/DifferentRescaleFactors/')

    # save hyperparameters in pickle file
    dic_hyp_par = {'directory_experiment': directory_experiment, 'name_experiment': name_experiment, 'directory_dataset': directory_dataset,
                   'name_dataset': name_dataset, 'test_size': test_size, 'validation_size': validation_size,
                   'max_num_epochs': max_num_epochs, 'max_epoch_constraints': max_epoch_constraints, 'batch_size': batch_size,
                   'initial_learning_rate': initial_learning_rate, 'acceptable_sat_ratio': acceptable_sat_ratio,
                   'max_bumps_con': max_bumps_con, 'max_time_after_bumps_con': max_time_after_bumps_con,
                   'max_constant_perf_con': max_constant_perf_con, 'max_restarts_con': max_restarts_con,
                   'dataset_states1': dataset_states1, 'dataset_states2': dataset_states2, 'weight_states': weight_states,
                   'optimizer': 'keras.optimizers.Adam',
                   'initializer': 'he_normal for weights of RELU layers, he_uniform for weights of linear layers, Constant for biases of RELU layers and Constant for biases of linear layers.'}

    with open(save_location_hyperparameters + '/DifferentRescaleFactors/' + 'Dictionary_hyperparameters.pickle', 'wb') as handle:
        pickle.dump(dic_hyp_par, handle, protocol=pickle.HIGHEST_PROTOCOL)

    list_bound_constraints, lower_bound_unnormalized, upper_bound_unnormalized = bcc.make_list_bound_constraints(margin=margin)

    # save the bound constraints used in pickle file
    dic_bound_con = {'list_bound_constraints': list_bound_constraints, 'lower_bound_unnormalized': lower_bound_unnormalized,
                     'upper_bound_unnormalized': upper_bound_unnormalized}

    with open(save_location_bound_constraints + '/DifferentRescaleFactors/' + 'Bound_constraints.pickle', 'wb') as handle:
        pickle.dump(dic_bound_con, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # gather stats from the training data, test data and training procedure
    input_dim = np.shape(input_data)[1]
    training_number = np.shape(input_data)[0]
    output_dim = np.shape(output_data)[1]

    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)

    previous_size_reductions = 0

    for r in rescale_factors:
        r_string = str(r)
        r = tf.constant(r, dtype=tf.float32)
        for n in range(0, len(train_size)):
            for k in range(0, len(dataset_states1)):
                # perform split of data into train, validation and test set
                x_train_complete, x_val, x_test, y_train_complete, y_val, y_test = split_train_test_val(input_data,
                                                                                                        output_data,
                                                                                                        test_size,
                                                                                                        validation_size,
                                                                                                        dataset_states1[k],
                                                                                                        dataset_states2[k])

                x_train, y_train = split_train(x_train_complete, y_train_complete, train_size[0:n], train_size[n], dataset_states3)

                # transform training data, validation data and test data to tf.data.Dataset
                train_dataset = make_dataset(x_train, y_train, batch_size)
                val_dataset = make_dataset(x_val, y_val, batch_size)
                test_dataset = make_dataset(x_test, y_test, batch_size)

                for i in range(0, len(weight_states)):
                    model = build_model_on_seed(custom_seed=weight_states[i], input_dim=input_dim, output_dim=output_dim)
                    model.compile(optimizer=optimizer)

                    if i == 0:
                        model.summary()

                        if constrained:
                            if not os.path.isdir(save_location_model + 'weights_con_model_' + str(i) + '_dataset_' + str(k) + '_size_' + str(train_size[n]) + '_rescale_factor_' + r_string):
                                os.makedirs(save_location_model + 'weights_con_model_' + str(i) + '_dataset_' + str(k) + '_size_' + str(train_size[n]) + '_rescale_factor_' + r_string)

                    if constrained:
                        model.save_weights(filepath=save_location_model + 'weights_con_model_' + str(i) + '_dataset_' + str(k) + '_size_' + str(train_size[n]) + '/weights_con_model_initial_' + str(i) + '_dataset_' + str(k) + '_size_' + str(train_size[n]) + '_rescale_factor_' + r_string)

                    # train the model with constraints
                    if constrained:
                        trained_model, train_loss_results, train_metric_results, train_satisfaction_results, val_loss_results, val_metric_results, val_satisfaction_results, epoch, list_lr, list_lr_loss, list_lr_con, timing, best_epoch = \
                            cggd.train_model_adaptive_rescale(model=model, input_dim=input_dim, output_dim=output_dim, optimizer=optimizer,
                                                              list_bound_constraints=list_bound_constraints,
                                                              train_number=np.size(x_train, 0), train_dataset=train_dataset,
                                                              train_batch_size=batch_size,  val_number=np.size(x_val, 0), val_dataset=val_dataset,
                                                              val_batch_size=batch_size, max_epochs=max_num_epochs, max_con_epochs=max_epoch_constraints,
                                                              rescale_constant=r, learning_rate=initial_learning_rate,
                                                              acceptable_sat_ratio=acceptable_sat_ratio, max_bumps=max_bumps_con,
                                                              max_time_after_bump=max_time_after_bumps_con, max_constant_perf=max_constant_perf_con,
                                                              max_restarts=max_restarts_con,
                                                              save_location_model=save_location_model + 'weights_con_model_' + str(i) + '_dataset_' + str(k) + '_size_' + str(train_size[n]) + '_rescale_factor_' + r_string,
                                                              save_file_model='/weights_con_model_trained_' + str(i) + '_dataset_' + str(k) + '_size_' + str(train_size[n]) + '_rescale_factor_' + r_string)

                    # test the constrained network on the test set
                    loss_value, metric_value, sat_ratio_value, rel_sat_ratio_value = \
                        cggd.test_model(model=trained_model, input_dim=input_dim, output_dim=output_dim,
                                       list_bound_constraints=list_bound_constraints, test_number=np.size(x_test, 0), test_dataset=test_dataset,
                                       test_batch_size=batch_size, acceptable_sat_ratio=acceptable_sat_ratio)

                    # save training results in pickle file
                    results = {'train_loss_results': train_loss_results, 'val_loss_results': val_loss_results,
                               'train_metric_results': train_metric_results, 'val_metric_results': val_metric_results,
                               'train_satisfaction_results': train_satisfaction_results,
                               'val_satisfaction_results': val_satisfaction_results, 'loss_value_test': loss_value,
                               'metric_value_test': metric_value, 'sat_ratio_test': sat_ratio_value, 'rel_sat_ratio_test': rel_sat_ratio_value,
                               'timing': timing, 'best_epoch': best_epoch}

                    with open(save_location_training + '_' + str(i) + '_dataset_' + str(k) + '_size_' + str(train_size[n]) + '_rescale_factor_' + r_string + '.pickle', 'wb') as handle:
                        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    if constrained:
                        results_lr = {'lr': list_lr, 'lr_loss': list_lr_loss, 'lr_con': list_lr_con}

                    with open(save_location_lr + '_' + str(i) + '_dataset_' + str(k) + '_size_' + str(train_size[n]) + '_rescale_factor_' + r_string + '.pickle', 'wb') as handle:
                        pickle.dump(results_lr, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # compute auxilary variables
                    x = range(1, len(train_loss_results) + 1)
                    if constrained:
                        x_lr = range(1, len(list_lr) + 1)

                    aux_var = {'epoch': epoch, 'x': x, 'x_lr': x_lr, 'train_size': np.size(x_train, 0), 'val_size': np.size(x_val, 0)}

                    with open(save_location_aux_variables + '_' + str(i) + '_dataset_' + str(k) + '_size_' + str(train_size[n]) + '_rescale_factor_' + r_string + '.pickle', 'wb') as handle:
                        pickle.dump(aux_var, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    if constrained:
                        print('--------------------------TEST RESULTS------------------')
                        print('Loss value constrained model: ' + str(loss_value) + '.')
                        print('Metric value constrained model: ' + str(metric_value) + '.')
                        print('Satisfaction ratio constrained model: ' + str(sat_ratio_value) + '.')
                        print('Relative satisfaction ratio constrained model: ' + str(rel_sat_ratio_value) + '.')
                        print('===================PART=OF=EXPERIMENT=ENDED=============')

    return


def perform_experiment_semi_sup_different_rescale_factors(directory_dataset, name_dataset, extension_dataset, initial_learning_rate, test_size,
                                                          validation_size, max_num_epochs, max_epoch_constraints, batch_size, acceptable_sat_ratio,
                                                          max_bumps_con, max_time_after_bumps_con, max_constant_perf_con, max_restarts_con,
                                                          random_state, number_dataset_realizations, number_weight_initializations, name_experiment,
                                                          margin, percentages_unsupervised, rescale_factors):
    """ Perform the experiment for the dataset at directory_experiment + name_experiment + extension_experiment.

    Nothing is returned since all results are saved into files.

    Parameters
    ----------
    directory_dataset : string
        String representing the directory where the dataset is stored.
    name_dataset : string
        String representing the name of the save file of the dataset.
    extension_dataset : string
        String representing the extension of the save file of the dataset.
    initial_learning_rate : float
        Float giving the learning rate used at the beginning of the training procedure.
    test_size : float
        Float in (0,1) indicating the size of the test set compared to the total data available.
    validation_size : float
        Float in (0,1) indicating the size of the validation set compared to the total data available.
    max_num_epochs : int
        Integer defining the maximum number of epochs that can be used to optimized the loss function only. The loss function is only
        optimized when all the examples in the train set satisfy the constraints. The training process is stopped when this number is
        reached.
    max_epoch_constraints : int
        Integer defining the maximum number of epochs that can be used to optimize the constraints. The training process is stopped when
        this number is reached.
    batch_size : int
        Integer defining the batch size used for training and validation.
    acceptable_sat_ratio : float
        Float between [0,1] indicating how large the satisfaction ratio needs to be in order to allow for training the loss function only.
        Setting this value to 1 will result in the procedure as described in the corresponding paper. Setting this value to 0 will result in
        training without constraints.
    max_bumps_con : int
        Integer denoting the maximal number of bumps that are allowed before reverting the model to a previous best found model.
    max_time_after_bumps_con : int
        Integer denoting the maximum number of epochs that can be used to obtain a better model after leaving the acceptable model space. If
        this number is reached, then the model is reverted to a previous best found model.
    max_constant_perf_con : int
        Integer denoting how long the algorithm can train without improving the performance. This is used as an early-stopping criterion.
    max_restarts_con : int
        Integer denoting the maximal number of reverting of the model that can be done before ending the training procedure.
    random_state : int
        Integer denoting the pseudo random seed used to initialize all the random values.
    number_dataset_realizations : int
        Integer denoting the amount of realizations need to be performed for the dataset.
    number_weight_initializations : int
        Integer denoting the amount of different network initializations need to be performed.
    name_experiment : string
        String containing the name of the experiment, which is used as a directory to store all the results.
    margin : tf.Tensor
        Constant tensor of shape=(,) and dtype=tf.float32 used to adjust the bound constraints extracted from the dataset.
    rescale_factors : list[float]
        List of floats (strictly larger than 1) denoting the different rescale factors.

    Returns
    ----------

    """
    constrained = True
    seed(random_state)
    ran_min = round(random() * 10)  # take random lower bound on the pseudo random seeds used
    ran_max = round(random() * (10 ** 6))  # take random upper bound on the pseudo random seeds used

    weight_states = []  # list containing the pseudo random seeds used to initialize the network weights
    dataset_states1 = []  # list containing the pseudo random seeds used to divide the data in a test set and an other set
    dataset_states2 = []  # list containing the pseudo random seeds used to divide the other set into a train set and a validation set.

    for i in range(0, number_weight_initializations):
        weight_states.append(randint(ran_min, ran_max))

    for i in range(0, number_dataset_realizations):
        dataset_states1.append(randint(ran_min, ran_max))
        dataset_states2.append(randint(ran_min, ran_max))

    supervision_pseudo_random_seed = randint(ran_min, ran_max)

    tf.random.set_seed(randint(ran_min, ran_max))

    input_data, output_data = prep_data(prep_directory=directory_dataset, prep_name=name_dataset, prep_ext=extension_dataset)

    directory_experiment = './Results'

    if not os.path.isdir(directory_experiment):
        os.makedirs(directory_experiment)

    if not os.path.isdir(directory_experiment + '/' + name_experiment):
        os.makedirs(directory_experiment + '/' + name_experiment)

    if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset):
        os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset)

    if constrained:  # Put all the constrained models in a directory and the unconstrained models in a separate one
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained')
        save_location_model = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Model'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Model'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Model')
        save_location_training = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/History')
        save_location_lr = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Learning_rate'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Learning_rate'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Learning_rate')
        save_location_aux_variables = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Aux_variables'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Aux_variables'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Aux_variables')
        save_location_hyperparameters = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Hyperparameters'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Hyperparameters'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Hyperparameters')
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Hyperparameters' + '/DifferentRescaleFactors/'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Hyperparameters' + '/DifferentRescaleFactors/')
        save_location_bound_constraints = directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Bound_constraints'
        if not os.path.isdir(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Bound_constraints'):
            os.makedirs(directory_experiment + '/' + name_experiment + '/' + name_dataset + '/Constrained/Bound_constraints')
        if not os.path.isdir(save_location_bound_constraints + '/DifferentRescaleFactors/'):
            os.makedirs(save_location_bound_constraints + '/DifferentRescaleFactors/')

    # save hyperparameters in pickle file
    dic_hyp_par = {'directory_experiment': directory_experiment, 'name_experiment': name_experiment, 'directory_dataset': directory_dataset,
                   'name_dataset': name_dataset, 'test_size': test_size, 'validation_size': validation_size,
                   'max_num_epochs': max_num_epochs, 'max_epoch_constraints': max_epoch_constraints, 'batch_size': batch_size,
                   'initial_learning_rate': initial_learning_rate, 'acceptable_sat_ratio': acceptable_sat_ratio,
                   'max_bumps_con': max_bumps_con, 'max_time_after_bumps_con': max_time_after_bumps_con,
                   'max_constant_perf_con': max_constant_perf_con, 'max_restarts_con': max_restarts_con,
                   'dataset_states1': dataset_states1, 'dataset_states2': dataset_states2, 'weight_states': weight_states,
                   'optimizer': 'keras.optimizers.Adam',
                   'initializer': 'he_normal for weights of RELU layers, he_uniform for weights of linear layers, Constant for biases of RELU layers and Constant for biases of linear layers.'}

    with open(save_location_hyperparameters + '/DifferentRescaleFactors/' + 'Dictionary_hyperparameters.pickle', 'wb') as handle:
        pickle.dump(dic_hyp_par, handle, protocol=pickle.HIGHEST_PROTOCOL)

    list_bound_constraints, lower_bound_unnormalized, upper_bound_unnormalized = bcc.make_list_bound_constraints(margin=margin)

    # save the bound constraints used in pickle file
    dic_bound_con = {'list_bound_constraints': list_bound_constraints, 'lower_bound_unnormalized': lower_bound_unnormalized,
                     'upper_bound_unnormalized': upper_bound_unnormalized}

    with open(save_location_bound_constraints + '/DifferentRescaleFactors/' + 'Bound_constraints.pickle', 'wb') as handle:
        pickle.dump(dic_bound_con, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # gather stats from the training data, test data and training procedure
    input_dim = np.shape(input_data)[1]
    training_number = np.shape(input_data)[0]
    output_dim = np.shape(output_data)[1]

    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)

    for rescale_factor in rescale_factors:
        rescale_factor_string = str(rescale_factor)
        rescale_factor = tf.constant(rescale_factor, dtype=tf.float32)

        for k in range(0, len(dataset_states1)):
            for r in range(0, len(percentages_unsupervised)):
                # perform split of data into train, validation and test set
                x_train, x_val, x_test, y_train, y_val, y_test = split_train_test_val(input_data,
                                                                                      output_data,
                                                                                      test_size,
                                                                                      validation_size,
                                                                                      dataset_states1[k],
                                                                                      dataset_states2[k])

                # transform training data, validation data and test data to tf.data.Dataset
                train_dataset = make_dataset(x_train, y_train, batch_size)
                val_dataset = make_dataset(x_val, y_val, batch_size)
                test_dataset = make_dataset(x_test, y_test, batch_size)

                number_of_batches = math.ceil(training_number / batch_size)
                supervised_batches = pick_supervised_batches(percentage_unsupervised=percentages_unsupervised[r],
                                                             number_of_batches=number_of_batches,
                                                             number_of_training_points=training_number,
                                                             batch_size=batch_size,
                                                             random_seed=supervision_pseudo_random_seed)

                for i in range(0, len(weight_states)):
                    model = build_model_on_seed(custom_seed=weight_states[i], input_dim=input_dim, output_dim=output_dim)
                    model.compile(optimizer=optimizer)

                    if i == 0:
                        model.summary()

                        if not os.path.isdir(save_location_model + 'weights_con_model_' + str(i) + '_dataset_' + str(k) + '_percentage_' + str(r) + '_rescale_factor_' + rescale_factor_string):
                            os.makedirs(save_location_model + 'weights_con_model_' + str(i) + '_dataset_' + str(k) + '_percentage_' + str(r) + '_rescale_factor_' + rescale_factor_string)

                    model.save_weights(filepath=save_location_model + 'weights_con_model_' + str(i) + '_dataset_' + str(k) + '_percentage_' + str(r) + '/weights_con_model_initial_' + str(i) + '_dataset_' + str(k) + '_percentage_' + str(r) + '_rescale_factor_' + rescale_factor_string)

                    # train the model with constraints
                    if constrained:
                        trained_model, train_loss_results, train_metric_results, train_satisfaction_results, val_loss_results, val_metric_results, val_satisfaction_results, epoch, list_lr, list_lr_loss, list_lr_con, timing, best_epoch = \
                            cggd.train_model_semi_sup(model=model, input_dim=input_dim, output_dim=output_dim, optimizer=optimizer,
                                                     list_bound_constraints=list_bound_constraints,
                                                     train_number=np.size(x_train, 0), train_dataset=train_dataset, train_batch_size=batch_size,
                                                     val_number=np.size(x_val, 0), val_dataset=val_dataset, val_batch_size=batch_size,
                                                     max_epochs=max_num_epochs, max_con_epochs=max_epoch_constraints, rescale_constant=rescale_factor,
                                                     learning_rate=initial_learning_rate, acceptable_sat_ratio=acceptable_sat_ratio,
                                                     max_bumps=max_bumps_con, max_time_after_bump=max_time_after_bumps_con,
                                                     max_constant_perf=max_constant_perf_con, max_restarts=max_restarts_con,
                                                     save_location_model=save_location_model + 'weights_con_model_' + str(i) + '_dataset_' + str(k) + '_percentage_' + str(r) + '_rescale_factor_' + rescale_factor_string,
                                                     save_file_model='/weights_con_model_trained_' + str(i) + '_dataset_' + str(k) + '_percentage_' + str(r) + '_rescale_factor_' + rescale_factor_string,
                                                     sup_batches=supervised_batches)

                    # test the constrained network on the test set
                    loss_value, metric_value, sat_ratio_value, rel_sat_ratio_value = \
                        cggd.test_model(model=trained_model, input_dim=input_dim, output_dim=output_dim,
                                       list_bound_constraints=list_bound_constraints, test_number=np.size(x_test, 0), test_dataset=test_dataset,
                                       test_batch_size=batch_size, acceptable_sat_ratio=acceptable_sat_ratio)

                    # save training results in pickle file
                    results = {'train_loss_results': train_loss_results, 'val_loss_results': val_loss_results,
                               'train_metric_results': train_metric_results, 'val_metric_results': val_metric_results,
                               'train_satisfaction_results': train_satisfaction_results,
                               'val_satisfaction_results': val_satisfaction_results, 'loss_value_test': loss_value,
                               'metric_value_test': metric_value, 'sat_ratio_test': sat_ratio_value, 'rel_sat_ratio_test': rel_sat_ratio_value,
                               'timing': timing, 'best_epoch': best_epoch}

                    with open(save_location_training + '_' + str(i) + '_dataset_' + str(k) + '_percentage_' + str(r) + '_rescale_factor_' + rescale_factor_string + '.pickle', 'wb') as handle:
                        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    if constrained:
                        results_lr = {'lr': list_lr, 'lr_loss': list_lr_loss, 'lr_con': list_lr_con}

                    with open(save_location_lr + '_' + str(i) + '_dataset_' + str(k) + '_percentage_' + str(r) + '_rescale_factor_' + rescale_factor_string + '.pickle', 'wb') as handle:
                        pickle.dump(results_lr, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # compute auxilary variables
                    x = range(1, len(train_loss_results) + 1)
                    if constrained:
                        x_lr = range(1, len(list_lr) + 1)

                    aux_var = {'epoch': epoch, 'x': x, 'x_lr': x_lr, 'train_size': np.size(x_train, 0), 'val_size': np.size(x_val, 0)}

                    with open(save_location_aux_variables + '_' + str(i) + '_dataset_' + str(k) + '_percentage_' + str(r) + '_rescale_factor_' + rescale_factor_string + '.pickle', 'wb') as handle:
                        pickle.dump(aux_var, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    if constrained:
                        print('--------------------------TEST RESULTS------------------')
                        print('Loss value constrained model: ' + str(loss_value) + '.')
                        print('Metric value constrained model: ' + str(metric_value) + '.')
                        print('Satisfaction ratio constrained model: ' + str(sat_ratio_value) + '.')
                        print('Relative satisfaction ratio constrained model: ' + str(rel_sat_ratio_value) + '.')
                        print('===================PART=OF=EXPERIMENT=ENDED=============')

    return

