from basic_model import TestDataConfig, TrainDataConfig

import numpy as np
import tensorflow as tf


def test_dataset(data_config, sess, batch_sizes):
    """
    Check that the data is correctly configured. Read in via tf.data.TextLineDataset
    """
    # Set all the batch sizes
    for batch_size in batch_sizes:
        data_config.batch_size = batch_size
        data_config.create_data_init()  # need to create_data_init() else using default batch_size

        sess.run(data_config.iter_init)
        next_batch = sess.run(data_config.next_element)
        # Check we are collected correct num of points
        for element in next_batch:
            assert element.shape[0] == batch_size
        print('Correct batch size received: got {got} and expected {expect}'.format(
            got=element.shape[0], expect=batch_size))

    # Check the typing for each datatype
    user, movie, rating, timestamp = next_batch
    assert user.dtype == np.int32
    assert movie.dtype == np.int32
    assert rating.dtype == np.float32
    assert timestamp.dtype == np.float32
    print('Correct datatypes received.')

    return True


if __name__ == '__main__':
    # DataConfigs to test
    test = TestDataConfig()
    train = TrainDataConfig()

    sess = tf.Session()
    result1 = test_dataset(test, sess, batch_sizes=[10, 100, 1000])
    result2 = test_dataset(train, sess, batch_sizes=[10, 100, 1000])

    if result1 and result2:
        print('All tests passed!')
