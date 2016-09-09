import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

import input_pipeline as ip
import model as m
from configuration import Configuration


if __name__ == "__main__":

    training_steps = int(sys.argv[1])

    file_names = ["data/data_batch_%d.bin" % i for i in range(1, 6)]

    pipeline = ip.DataPipeline(file_names)
    train_x, train_y = pipeline.get_batch_op()

    config = Configuration(
        input_size=3072,
        output_size=10,
        batch_size=100,
        num_epoch=1000000
    )

    global_step = tf.Variable(0, trainable=False)

    convo_model = m.ConvoModel(train_x, train_y, config)
    convo_model.initialize()

    loss_op, train_op = convo_model.train_op(global_step)

    with tf.Session() as session:

        init = tf.initialize_all_variables()
        merged = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter("summary/", session.graph)

        tf.train.start_queue_runners(session)
        session.run(init)

        for i in range(training_steps):

            start_time = time.time()
            loss, _ = session.run([loss_op, train_op])
            duration = time.time() - start_time

            assert not np.isnan(loss), "Model diverged with loss=NaN"

            if i % 10 == 0:
                
                num_examples_per_step = config.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = "%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)"

                print(format_str % (datetime.now(), i, loss, examples_per_sec, sec_per_batch))

            if i % 100 == 0:
                
                summary_str = session.run(merged)
                summary_writer.add_summary(summary_str, i)
