# Keras multiplexer
This is a Keras layer that acts as a multiplexer for `Dense` layers (or
any other layer that has 1D output).

This layer is used to split the output of the previous layer into
N groups of size `output_dim`, and choose which group to activate
as output using a discrete control signal.
During training, the only weights that are updated are those of the
active group, while the others remain unchanged.

![mux_1_w](https://cloud.githubusercontent.com/assets/11634240/26721634/d8aed964-478c-11e7-930b-fef27eb36fe3.png)

![mux_2_w](https://cloud.githubusercontent.com/assets/11634240/26721706/22f5cfdc-478d-11e7-8eaa-51b6a3101328.png)

The layer takes as input two tensors, namely the output of the previous
layer and a column tensor of type `int32` or `int64` for the control
signal.

The input to `Multiplexer` (i.e. the output of the previous layer) must be
of shape `(None, N * output_dim)`, and the values in the control tensor
must be between 0 (inclusive) and N (exclusive).

**No checks are done at runtime to ensure that the input to the layer is
correct or that the control signal contains legal values, so it's
better to double check.**

While basically implementing a controlled version of `Dropout`, this
layer can be especially useful when learning a multidimensional function
associated to a discrete space that conditions the output.

An example of this is deep Q-learning, where the Q-function depends on
an action that can be discrete.
In [the DQN paper by DeepMind](https://arxiv.org/abs/1312.5602), the
Q-network is trained by setting the target to be equal to the network
output on all actions except the one being updated, as follows:
```
# I'm not using any particular notation
for sample in batch:
    # ...
    target = q_network.predict(sample.state)
    target[sample.action] = sample.reward +
                            df * max(q_network.predict(sample.state_))
    # ...
q_network.fit(states, targets)
```

This requires an extra forward pass of the Q-network in order to compute
the target, which is not really necessary.
With the `Multiplexer` layer, the same result can be achieved by simply
feeding the action to the network as a separate input and updating only
the associated weights (see example below for details on implementation).

# Installation
It seemed overkill to package this as a library, so just copy and paste
`multiplexer.py` in your project to use it.
**Note that the layer only works with `Keras>=2.0.0` and the Tensorflow
backend.**

## Example
This example implements the NN represented in the images above.
```py
from numpy import array
from numpy.random import randn
from keras.models import Model
from keras.layers import Input, Dense
from multiplexer import Multiplexer

# Model definition
input = Input(shape=(3,))
control = Input(shape=(1,), dtype='int32')
hidden = Dense(6)(i)  # output_dim == 2, nb_ctrl_sig == 3
output = Multiplexer(2, 3)([hidden, control])

# Build and compile model
model = Model(input=[input, control], output=output)
model.compile('sgd', 'mse')

# Data
x = randn(3)  # Input has size 3
ctrl = array([0, 1, 2])

# Outputs the first two neurons of the Dense layer
model.predict([x, ctrl[0]])

# Outputs the middle two neurons of the Dense layer
model.predict([x, ctrl[1]])

# Outputs the last two neurons of the Dense layer
model.predict([x, ctrl[2]])
```

To adapt this example to the DQN case, we would use two different
models (`q_net_train` for training and `q_net_test` for testing) 
respectively with output layers `output` and `hidden`, and the 
`Multiplexer` layer configured with `output_dim == 1` and 
`nb_ctrl_sig == 6`.  
We could then use `sample.reward + df * max(q_net_test.predict(sample.state_))` 
as single target, and pass `sample.state` and `sample.action` as input to
`q_net_train`.

## Acknowledgments
Thanks to @carloderamo for porting the previous implementation to Keras 2.
