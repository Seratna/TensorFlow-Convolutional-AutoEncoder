import tensorflow as tf


class Model(object):
    """

    """

    def __init__(self):
        pass

    @staticmethod
    def start_new_session(sess):
        saver = tf.train.Saver()  # create a saver
        global_step = 0

        sess.run(tf.global_variables_initializer())
        print('started a new session')

        return saver, global_step

    @staticmethod
    def continue_previous_session(sess, ckpt_file):
        saver = tf.train.Saver()  # create a saver

        with open(ckpt_file) as file:  # read checkpoint file
            line = file.readline()  # read the first line, which contains the file name of the latest checkpoint
            ckpt = line.split('"')[1]
            global_step = int(ckpt.split('-')[1])

        # restore
        saver.restore(sess, 'saver/'+ckpt)
        print('restored from checkpoint ' + ckpt)

        return saver, global_step


def main():
    pass


if __name__ == '__main__':
    main()
