import tensorflow as tf


LABEL_SIZE = 1


class DataPipeline(object):

    def __init__(
            self,
            file_names,
            image_w=32,
            image_h=32,
            num_labels=10,
            batch_size=100
            ):

        self._data_size = image_w * image_h * 3
        self._total_size = self._data_size + LABEL_SIZE

        self._w = image_w
        self._h = image_h

        self._num_labels = num_labels
        self._batch_size = batch_size
        self._file_names = file_names

    def get_batch_op(self):

        self._file_name_queue = tf.train.string_input_producer(
           self._file_names
        )

        x, y = self._get_raw_data_op(self._file_name_queue)

        return self._get_batch_op(x, y)

    def _get_raw_data_op(self, file_name_queue):

        self._file_reader = tf.FixedLengthRecordReader(self._total_size)

        (raw_key, raw_value) = self._file_reader.read(file_name_queue)

        uint8_value = tf.decode_raw(raw_value, tf.uint8)

        y = tf.squeeze(tf.slice(uint8_value, [0], [LABEL_SIZE]))
        x = tf.slice(uint8_value, [LABEL_SIZE], [self._data_size])

        return self._process_image(tf.cast(x, tf.float32)), tf.cast(y, tf.int64)

    def _process_image(self, x):

        reshape_x = tf.transpose(
            tf.reshape(x, [3, self._h, self._w]),
            (1, 2, 0)
        )

        return tf.image.per_image_whitening(reshape_x)

    def _get_batch_op(self, x, y):

        x, y = tf.train.shuffle_batch(
                            [x, y],
                            batch_size=self._batch_size,
                            capacity=self._batch_size*3,
                            min_after_dequeue=self._batch_size*2
                        )

        return x, y
