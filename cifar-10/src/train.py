import tensorflow as tf
import input_pipeline as ip
import model as m
from configuration import Configuration


if __name__ == "__main__":

    file_names = ["data/data_batch_%d.bin" % i for i in range(1, 6)]

    pipeline = ip.DataPipeline(file_names)
    train_x, train_y = pipeline.get_batch_op()

    config = Configuration(
        input_size=3072,
        output_size=10,
        batch_size=100,
        num_epoch=1000000
    )

    convo_model = m.ConvoModel(train_x, train_y, config)
    convo_model.initialize()

    train_op = convo_model.train_op()

    with tf.Session() as session:

        init = tf.initialize_all_variables()

        tf.train.start_queue_runners(session)
        session.run(init)

        for i in range(100):

            session.run(train_op)
