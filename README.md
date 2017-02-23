# TensorFlow Convolutional AutoEncoder

This is an implementation of Convolutional AutoEncoder using only TensorFlow.


## Implementation

### Model structure

The structure of this conv autoencoder is shown below:

![autoencoder structure](https://cloud.githubusercontent.com/assets/13087207/23150628/cd447882-f7c2-11e6-938a-b8e672d71760.png)

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

Training was done using GTX1070 GPU, batch size 100, 50000 passes.

Trained weights of the 1st convolutional layer are shown below, 32 of 5x5 kernels:
![weights](https://cloud.githubusercontent.com/assets/13087207/22701950/20c88bce-ed2d-11e6-8b0b-fd9e782c2680.png)

And here's some of the reconstruction results:

![reconstruction_1](https://cloud.githubusercontent.com/assets/13087207/22701953/20d2519a-ed2d-11e6-9f5d-4602ca1459bb.png)
![reconstruction_2](https://cloud.githubusercontent.com/assets/13087207/22701952/20c93b78-ed2d-11e6-9b6c-c66ccc8a8200.png)
![reconstruction_3](https://cloud.githubusercontent.com/assets/13087207/22701951/20c8f6c2-ed2d-11e6-9594-c3f3b370eb38.png)
