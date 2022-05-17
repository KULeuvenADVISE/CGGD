Overview Files
===

In order to run the experiment, run the file __main.py__. The file __boundConstraints.py__ contains the bound constraints used during the experiment. The file __aux.py__ converts the dataset to the desired object, constructs the network and performs the experiment (all the necessary directories are made to save the results). The file __cggdbias.py__ contains the implementation of CGGD for the Bias Correction data set and with the constraint that the minimal temperature of the next day is smaller or equal than the maximal temperature of the next day. The latter constraint will be referred to as the __sum constraint__.


CGGD Implementation
===
The comments in this section are relevant for the __cggdbias.py__ file.


The main functions to train and check the performance of the model are: __train_model__, __train_model_unconstrained__ and __test_model__. This implementation is a proof of concept and there are many places where the code can be optimized in terms of speed and memory usage.

The other functions in this file are auxilary functions for computing the satisfaction of a constraint or the gradient for a constraint if it is not satisfied.

__train_model__
---
This function trains the model with constraints as described in CGGD. The constraint module and the automatic differentiation is done by the function __compute_sum_grad__. This function computes the gradient of the loss function, the gradient of the bound constraints and the gradient of the sum constraint. Moreover, the function also computes the value of the loss function and the SR of the current batch of data points. The function __rescale_sum_grad__ corresponds to the processing gradient module. This function combines the gradient of the loss function and the rescaled gradients of the constraints. The rescaling itself is also performed in this function. The result is the gradient that can be used by the optimizer to update the model. It is chosen to normalize and rescale over the rows of the weight matrices and over the biases individually.

The function __compute_sum_grad__ is structured as follows. 
1. Compute forward pass
2. Compute value of the loss function
3. Compute the satisfaction of the constraints
4. Compute the gradient on the variables present in the constraints that are not satisfied
5. Compute (with backpropagation) the gradient of the loss function and the gradient of the unsatisfied constraints corresponding to the gradient computed in step 4.

The bound constraints are grouped together according to their type (i.e. smaller or equal, strictly smaller, larger or equal, and strictly larger). 

The checking of the constraints is done in the __compute_sum_grad__ function on line 1204. More specific, in this function the checking is done on the lines 676-716. The computation of the direction on the output variables is done on the same lines. In other words, the __constraint module__ is implemented on these lines 676-716. The __processing gradient module__ consists of the lines 744-753 and the function __rescale_sum_grad__ on line 1223. In the first set of lines the computation of the gradient of the constraints is done, i.e. the direction on the output variables is propagated to the network by doing a backward pass. The normalization and rescaling is done by the function __rescale_sum_grad__. The __adaptive learning rate schedule__ corresponds to the lines 1320-1468. 

Note that the training function consists mainly about a duplication of the framework. The reason for this is that the last batch of data could have a different batch size (although not in case of the performed experiments). Many auxilary tf.Tensor s are dependent on this batch size and thus different variables are used for the final batch if this batch has a different size.


__train_model_unconstrained__
---
This function trains the model without constraints and is thus very similar to the function __fit__ in TensorFlow. The only difference is that the constraints are checked for satisfaction during each iteration in order to be able to compute the Satisfaction Ratio (SR) and compare with the model that is trained by CGGD.

__test_model__
---

This function can be used to compute the performance (MSE and SR) for a given model and a given test set.






