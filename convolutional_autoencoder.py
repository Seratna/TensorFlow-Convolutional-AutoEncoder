import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from mnist import MNIST  # this is the MNIST data manager that provides training/testing batches


class ConvolutionalAutoencoder(object):
    """

    """
    def __init__(self):
        """
        build the graph
        """
        IMG_HEIGHT = 28  # each image of a digit is 28x28
        IMG_WIDTH = 28
        NUM_CHANNELS = 1  # grey scale (only 1 channel)

        # place holder of input data and label
        x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH])

        # reshape each image to have 1 channel
        x_image = tf.reshape(x, [-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])  # [#batch, img_height, img_width, #channels]

        # Convolutional Layer 1
        with tf.variable_scope('conv_1') as scope:
            conv1 = self.conv_layer(x_image, [5, 5, NUM_CHANNELS, 32])

        # max pooling layer 1
        with tf.variable_scope('pool_1') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolutional Layer 2
        with tf.variable_scope('conv_2') as scope:
            conv2 = self.conv_layer(pool1, [5, 5, 32, 32])

        # max pooling
        with tf.variable_scope('pool_2') as scope:
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (-1, 7, 7, 32)

        # un-pooling
        with tf.variable_scope('unpool_1') as scope:
            unpool1 = self.unpooling_layer(pool2, kernel_shape=(2, 2), output_shape=tf.shape(conv2))

        # Deconvolution (transpose of conv2d)
        with tf.variable_scope('deconv_1') as scope:
            deconv1 = self.deconv_layer(unpool1, (5, 5, 32, 32), output_shape=tf.shape(pool1))

        # un-pooling
        with tf.variable_scope('unpool_2') as scope:
            unpool2 = self.unpooling_layer(deconv1, kernel_shape=(2, 2), output_shape=tf.shape(conv1))

        # Deconvolution (transpose of conv2d)
        with tf.variable_scope('deconv_2') as scope:
            reconstruction = self.deconv_layer(unpool2, (5, 5, 1, 32), output_shape=tf.shape(x_image))

        # loss function
        loss = tf.nn.l2_loss(x_image - reconstruction)  # L2 loss

        # training
        training = tf.train.AdamOptimizer(1e-4).minimize(loss)

        #
        self.x = x

        self.x_image = x_image
        self.reconstruction = reconstruction
        self.loss = loss
        self.training = training

    def train(self, batch_size, passes, new_training=True):
        """

        :param batch_size:
        :param passes:
        :param new_training:
        :return:
        """
        mnist = MNIST()
        saver = tf.train.Saver()  # create a saver
        global_step = 0

        with tf.Session() as sess:
            # prepare session
            if new_training:  # start a new training session
                sess.run(tf.global_variables_initializer())
                print('started new session')
            else:  # resume from a previous training session
                with open('saver/checkpoint') as file:  # read checkpoint file
                    line = file.readline()  # read the first line, which contains the file name of the latest checkpoint
                    ckpt = line.split('"')[1]
                    global_step = int(ckpt.split('-')[1])
                # restore
                saver.restore(sess, 'saver/'+ckpt)
                print('restored from checkpoint ' + ckpt)

            # start training
            for step in range(1+global_step, 1+passes+global_step):
                x, y = mnist.get_batch(batch_size)
                self.training.run(feed_dict={self.x: x})

                if step % 10 == 0:
                    loss = self.loss.eval(feed_dict={self.x: x})
                    print("pass {}, training loss {}".format(step, loss))

                if step % 1000 == 0:  # save weights
                    saver.save(sess, 'saver/cnn', global_step=step)
                    print('checkpoint saved')

    def reconstruct(self):
        """
        """

        def weights_to_grid(weights, rows, cols):
            """convert the weights tensor into a grid for visualization"""
            height, width, in_channel, out_channel = weights.shape
            padded = np.pad(weights, [(1, 1), (1, 1), (0, 0), (0, rows * cols - out_channel)],
                            mode='constant', constant_values=0)
            transposed = padded.transpose((3, 1, 0, 2))
            reshaped = transposed.reshape((rows, -1))
            grid_rows = [row.reshape((-1, height + 2, in_channel)).transpose((1, 0, 2)) for row in reshaped]
            grid = np.concatenate(grid_rows, axis=0)

            return grid.squeeze()

        mnist = MNIST()
        saver = tf.train.Saver()  # create a saver

        with tf.Session() as sess:
            with open('saver/checkpoint') as file:  # read checkpoint file
                line = file.readline()  # read the first line, which contains the file name of the latest checkpoint
                ckpt = line.split('"')[1]
            # restore
            saver.restore(sess, 'saver/'+ckpt)
            print('restored from checkpoint ' + ckpt)

            batch_size = 5
            x, y = mnist.get_batch(batch_size, dataset='testing')
            org, recon = sess.run((self.x_image, self.reconstruction), feed_dict={self.x: x})

            # visualize weights
            first_layer_weights = tf.get_default_graph().get_tensor_by_name("conv_1/weights:0").eval()
            grid = weights_to_grid(first_layer_weights, 8, 4)
            plt.imshow(grid, cmap=plt.cm.gray, interpolation='nearest')
            plt.title('first conv layers weights')
            plt.show()

            for i in range(batch_size):
                im = np.concatenate((org[i, :, :, :].reshape(28, 28), recon[i, :, :, :].reshape(28, 28)), axis=1)
                plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
                plt.text(0, 1, 'input img', color='w')
                plt.text(28, 1, 'reconstruction', color='w')
                plt.show()

    def conv_layer(self,
                   input_matrix,
                   weights_shape,
                   weights_init_stddev=0.1,
                   bias_init_value=0.1,
                   conv_strides=(1, 1, 1, 1),
                   conv_padding='SAME'):
        """
        build a convolutional layer.

        :param input_matrix:
        :param weights_shape: [window_height, window_width, #input_channels, #output_channels]
        :param weights_init_stddev:
        :param bias_init_value:
        :param conv_strides:
        :param conv_padding:
        """
        window_height, window_width, num_input_channels, num_output_channels = weights_shape

        weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=weights_init_stddev), name='weights')
        bias = tf.Variable(tf.constant(bias_init_value, shape=[num_output_channels]),
                           name='bias')  # 1 bias for each output channel
        conv = tf.nn.conv2d(input_matrix, weights, strides=conv_strides, padding=conv_padding)
        relu = tf.nn.relu(conv + bias)

        return relu

    def fully_connected_layer(self,
                              input_matrix,
                              weights_shape,
                              weights_init_stddev=0.1,
                              bias_init_value=0.1):
        """
        build a fully connected layer.

        :param input_matrix:
        :param weights_shape: [input_length, num_neurons]
        :param weights_init_stddev:
        :param bias_init_value:
        :return:
        """
        input_length, num_neurons = weights_shape
        weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=weights_init_stddev), name='weights')
        bias = tf.Variable(tf.constant(bias_init_value, shape=[num_neurons]), name='bias')
        fc = tf.matmul(input_matrix, weights) + bias

        return fc

    def unpooling_layer(self,
                        input_matrix,
                        kernel_shape,
                        output_shape):
        """
        Unpool a max-pooled layer.

        Currently this method does not use the argmax information from the previous pooling layer.
        Currently this method assumes that the size of the max-pooling filter is same as the strides.

        Each entry in the pooled map would be replaced with an NxN kernel with the original entry in the upper left.
        For example: a 1x2x2x1 map of

            [[[[1], [2]],
              [[3], [4]]]]

        could be unpooled to a 1x4x4x1 map of

            [[[[ 1.], [ 0.], [ 2.], [ 0.]],
              [[ 0.], [ 0.], [ 0.], [ 0.]],
              [[ 3.], [ 0.], [ 4.], [ 0.]],
              [[ 0.], [ 0.], [ 0.], [ 0.]]]]

        :param output_shape:
        :param input_matrix:
        :param kernel_shape:
        :return:
        """
        num_channels = input_matrix.get_shape()[-1]
        input_dtype_as_numpy = input_matrix.dtype.as_numpy_dtype()
        kernel_rows, kernel_cols = kernel_shape

        # build kernel
        kernel_value = np.zeros((kernel_rows, kernel_cols, num_channels, num_channels), dtype=input_dtype_as_numpy)
        kernel_value[0, 0, :, :] = np.eye(num_channels, num_channels)
        kernel = tf.constant(kernel_value)

        # do the un-pooling using conv2d_transpose
        unpool = tf.nn.conv2d_transpose(input_matrix,
                                        kernel,
                                        output_shape=output_shape,
                                        strides=(1, kernel_rows, kernel_cols, 1),
                                        padding='VALID')
        return unpool

    def deconv_layer(self,
                     input_matrix,
                     weights_shape,
                     output_shape,
                     weights_init_stddev=0.1,
                     bias_init_value=0.1,
                     conv_strides=(1, 1, 1, 1),
                     conv_padding='SAME'):
        """
        build a de-convolution layer.

        using tf.nn.conv2d_transpose() method.
        see: U{https://www.tensorflow.org/api_docs/python/nn/convolution#conv2d_transpose}

        :param input_matrix:
        :param weights_shape:
        :param output_shape:
        :param weights_init_stddev:
        :param bias_init_value:
        :param conv_strides:
        :param conv_padding:
        :return:
        """
        window_height, window_width, num_output_channels, num_input_channels = weights_shape

        weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=weights_init_stddev), name='weights')
        bias = tf.Variable(tf.constant(bias_init_value, shape=[num_output_channels]),
                           name='bias')  # 1 bias for each output channel
        deconv = tf.nn.conv2d_transpose(input_matrix,
                                        weights,
                                        output_shape=output_shape,
                                        strides=conv_strides,
                                        padding=conv_padding)
        relu = tf.nn.relu(deconv + bias)

        return relu


def main():
    conv_autoencoder = ConvolutionalAutoencoder()
    # conv_autoencoder.train(batch_size=100, passes=50000, new_training=True)
    conv_autoencoder.reconstruct()

if __name__ == '__main__':
    main()
