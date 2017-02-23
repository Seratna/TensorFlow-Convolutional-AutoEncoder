from abc import ABCMeta, abstractmethod

import tensorflow as tf


class Layer(object, metaclass=ABCMeta):
    """

    """
    def __init__(self):
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.call()


class Convolution2D(Layer):
    """

    """
    def __init__(self,
                 kernel_shape,
                 kernel=None,
                 bias=None,
                 strides=(1, 1, 1, 1),
                 padding='SAME',
                 activation=None):
        Layer.__init__(self)

        # build kernel
        if kernel:
            assert kernel.get_shape() == kernel_shape
            self.kernel = kernel
        else:
            self.kernel = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name='kernel')

        # build bias
        kernel_height, kernel_width, num_input_channels, num_output_channels = self.kernel.get_shape()
        if bias:
            assert bias.get_shape() == (num_output_channels, )
            self.bias = bias
        else:
            self.bias = tf.Variable(tf.constant(0.1, shape=[num_output_channels]), name='bias')

        self.strides = strides
        self.padding = padding
        self.activation = activation

    def call(self, input_tensor):
        conv = tf.nn.conv2d(input_matrix, weights, strides=conv_strides, padding=conv_padding)
        relu = tf.nn.relu(conv + bias)


def main():
    conv = Convolution2D([5, 5, 1, 32])


if __name__ == '__main__':
    main()
