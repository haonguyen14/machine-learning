import tensorflow as tf
import input_pipeline as ip
import model as m
from configuration import Configuration


def build_model(session, x, exp_name, a_function):

    config = Configuration(
        input_size=3072,
        output_size=10,
        examples_per_epoches=50000,
        batch_size=100,
        a_function=a_function
    )

    dropout = tf.placeholder(tf.float32)

    model = m.ConvoModel(config)
    model.initialize(x, dropout)

    saver = tf.train.Saver(tf.all_variables())

    checkpoint_path = tf.train.latest_checkpoint("experiments/%s" % exp_name)
    saver.restore(session, checkpoint_path)

    return model
