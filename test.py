import scipy.io as sio
import numpy as np
import theano.tensor as T
from theano import shared

from convnet3d import ConvLayer, NormLayer, PoolLayer, RectLayer
from mlp import LogRegr, HiddenLayer, DropoutLayer
from activations import relu, tanh, sigmoid, softplus

dataReadyForCNN = sio.loadmat("DataReadyForCNN.mat")

xTrain = dataReadyForCNN["xTrain"].astype('float64')
# xTrain = np.random.rand(10, 1, 5, 6, 2).astype('float64')
# xTrain.dtype

dtensor5 = T.TensorType('float64', (False,)*5)
x = dtensor5('x') # the input data

yCond = T.ivector()

# input = (nImages, nChannel(nFeatureMaps), nDim1, nDim2, nDim3)

kernel_shape = (5,6,2)
fMRI_shape = (51, 61, 23)
n_in_maps = 1 # channel
n_out_maps = 5 # num of feature maps, aka the depth of the neurons
num_pic = 2592

layer1_input = x
print layer1_input.eval({x:xTrain}).shape

convLayer1 = ConvLayer(layer1_input, n_in_maps, n_out_maps, kernel_shape, fMRI_shape, 
                       num_pic, tanh)

f = theano.function([x], 2*x)

print convLayer1.output.eval({x:xTrain}).shape