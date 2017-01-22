from __future__ import print_function

import scipy.io as sio
import numpy as np
import theano.tensor as T
import theano
from theano import shared
from lasagne.layers import InputLayer, DenseLayer

theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
# theano.config.compute_test_value = 'warn'

import os
import sys
import timeit

from mlp import LogRegr, HiddenLayer, DropoutLayer
from convnet3d import ConvLayer, NormLayer, PoolLayer, RectLayer
from activations import relu, tanh, sigmoid, softplus

# Get data

data = sio.loadmat("lowVarEliminatedData.mat")

xTrain = data["xTrain"].astype("float64")
yTrainCond = data["yTrainCond"].astype("int32")
yTrainWord = data["yTrainWord"].astype("int32")

xValidate = data["xValid"].astype("float64")
yValidateCond = data["yValidCond"].astype("int32")
yValidateWord = data["yValidWord"].astype("int32")

xTest = data["xTest"].astype("float64")
yTestCond = data["yTestCond"].astype("int32")
yTestWord = data["yTestWord"].astype("int32")


##################################
# Build Model
#################################
dtensor5 = T.TensorType('float64', (False,)*5)
x = dtensor5('x') # the input data
y = T.ivector()

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch

# input = (nImages, nChannel(nFeatureMaps), nDim1, nDim2, nDim3)

# layer1 (500, 5, 47, 56, 22)
# layer2 (500, 5, 10, 12, 5)
# layer3 (500, 3, 9, 11, 4)
# layer4 (500, 3, 5, 6, 2)

fMRI_shape = (51, 61, 23)

batch_size = 200

# 1st: Convolution Layer
layer1_input = x
layer1 = ConvLayer(layer1_input, 1, 10, (5, 5, 5), fMRI_shape, 
                       batch_size, sigmoid)

# print(layer1.output.eval({x:xTrain[:50]}).shape)

# layer1.output.eval({x:xTrain[1]}).shape[3:]


# 2nd: Pool layer
poolShape = (2, 2, 2)
layer2 = PoolLayer(layer1.output, poolShape)

# print(layer2.output.eval({x:xTrain[:50]}).shape)

# 3rd: Convolution Layer
layer3 = ConvLayer(layer2.output, 10, 10, (5, 5, 5), (24, 29, 10), 
                       batch_size, sigmoid)

# print(layer3.output.eval({x:xTrain[:50]}).shape)

# # 4th: Pool layer
layer4 = PoolLayer(layer3.output, (2, 2, 2))

# print(layer4.output.eval({x:xTrain[:50]}).shape)

# 5th: Dense layer
layer5_input = T.flatten(layer4.output, outdim=2)
layer5 = HiddenLayer(layer5_input, n_in=10*10*13*3, n_out=100, activation=sigmoid)

# print(layer5.output.eval({x:xTrain[:50]}).shape)

# 6th: Logistic layer
layer6 = LogRegr(layer5.output, 100, 12, sigmoid)

cost = layer6.negative_log_likelihood(y)

# create a function to compute the mistakes that are made by the model
test_model = theano.function(
    [index],
    layer6.errors(y),
    givens={
        x: shared(xTest)[index * batch_size: (index + 1) * batch_size],
        y: shared(yTestCond[0])[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    [index],
    layer6.errors(y),
    givens={
        x: shared(xValidate)[index * batch_size: (index + 1) * batch_size],
        y: shared(yValidateCond[0])[index * batch_size: (index + 1) * batch_size]
    }
)

# create a list of all model parameters to be fit by gradient descent
params = layer5.params + layer3.params + layer1.params + layer6.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i], grads[i]) pairs.
learning_rate=0.1

updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: shared(xTrain)[index * batch_size: (index + 1) * batch_size],
        y: shared(yTrainCond[0])[index * batch_size: (index + 1) * batch_size]
    }
)

###############
# TRAIN MODEL #
###############
import timeit


print('... training')

n_train_batches = xTrain.shape[0]
n_test_batches = xTest.shape[0]
n_valid_batches = xValidate.shape[0]

n_train_batches //= batch_size
n_valid_batches //= batch_size
n_test_batches //= batch_size

n_epochs=200

# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience // 2)
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set; in this case we
                              # check every epoch

best_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):

        minibatch_avg_cost = train_model(minibatch_index)

        print("minibatch_avg_cost calc of " + str(minibatch_index) + " done")

        # iteration number
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = []
            for i in xrange(n_valid_batches):
            	validation_losses.append(validate_model(i))

            # validation_losses = [validate_model(i) for i
            #                      in range(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)

            print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                )
            )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if (
                    this_validation_loss < best_validation_loss *
                    improvement_threshold
                ):
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = []
            	for i in xrange(n_test_batches):
					test_losses.append(test_model(i))
                
                test_score = np.mean(test_losses)

                print(('     epoch %i, minibatch %i/%i, test error of '
                       'best model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))

        # if patience <= iter:
        #     done_looping = True
        #     break

            
end_time = timeit.default_timer()
print(('Optimization complete. Best validation score of %f %% '
       'obtained at iteration %i, with test performance %f %%') %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
# print(('The code for file ' +
#        os.path.split(__file__)[1] +
#        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)