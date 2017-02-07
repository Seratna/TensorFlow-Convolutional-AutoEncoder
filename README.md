# TensorFlow Convolutional AutoEncoder

This is an implementation of Convolutional AutoEncoder using only TensorFlow.


## Implementation

### Model structure

The structure of this conv autoencoder is shown below:

![autoencoder structure](https://cloud.githubusercontent.com/assets/13087207/22667671/d9d22190-ec8b-11e6-84a9-0762621a3271.png)

This implementation was build for the MNIST dataset, but it is very easy to change the structure of the model
for other applications.

### Un-pooling

Since the max-pooling operation is not injective, and TensorFlow does not have a built-in unpooling method,
we have to implement our own approximation.
But it is actually easy to do so using TensorFlow's [`tf.nn.conv2d_transpose()`](https://www.tensorflow.org/api_docs/python/nn/convolution#conv2d_transpose) method.

The idea was to replace each entry in the pooled map with an NxM kernel with the original entry in the upper left,
where N and M are the shape of the pooling kernel.

![un-pooling](https://cloud.githubusercontent.com/assets/13087207/22672037/77e521c6-ec9f-11e6-9aba-119f954cd9f8.png)

This is equivalent to doing transpose of conv2d on the input map 
with a kernel that has 1 on the upper left and 0 elsewhere. 
Therefore we could do this trick with `tf.nn.conv2d_transpose()` method.

## Experiments


