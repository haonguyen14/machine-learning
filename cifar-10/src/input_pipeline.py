import tensorflow as tf
import cPickle


LABEL_SIZE = 1


class DataPipeline(object):

    def __init__(
            self,
            file_names,
            data_size=3072,
            batch_size=100
            ):
        
        self._total_size = data_size + LABEL_SIZE 
        self._data_size = data_size
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

        y = tf.slice(uint8_value, [0], [LABEL_SIZE])
        x = tf.slice(uint8_value, [LABEL_SIZE], [self._data_size]) 

        return x, y

    
    def _get_batch_op(self, x, y):

        return tf.train.shuffle_batch(
                            [x, y],
                            batch_size=self._batch_size,
                            capacity=self._batch_size*3,
                            min_after_dequeue=self._batch_size*2 
                        )
