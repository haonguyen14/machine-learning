import sys
import time
import os
from datetime import datetime

import tensorflow as tf
import numpy as np

import input_pipeline as ip
import model as m
from configuration import Configuration


if __name__ == "__main__":

    exp_name = sys.argv[1]
    training_steps = int(sys.argv[2])
    a_function = tf.nn.relu if sys.argv[3] == "relu" else tf.sigmoid

    exp_path = "experiments/%s" % exp_name
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    else:
        print("Experiment %s already exists" % exp_name)
        sys.exit()

    file_names = ["data/data_batch_%d.bin" % i for i in range(1, 6)]

    pipeline = ip.DataPipeline(file_names)
    train_x, train_y = pipeline.get_batch_op()

    config = Configuration(
        input_size=3072,
        output_size=10,
        batch_size=100,
        num_epoch=1000000,
        a_function=a_function
    )

    global_step = tf.Variable(0, trainable=False)

    convo_model = m.ConvoModel(config)
    convo_model.initialize(train_x)

    loss_op, train_op = convo_model.train_op(train_y, global_step)

    with tf.Session() as session:

        init = tf.initialize_all_variables()
        merged = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(exp_path, session.graph)

        saver = tf.train.Saver(tf.all_variables())

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

                format_str = "%s: step %d, loss = %.2f"
                format_str += " (%.2f examples/sec; %.3f sec/batch)"

                print(
                    format_str % (
                            datetime.now(),
                            i,
                            loss,
                            examples_per_sec,
                            sec_per_batch)
                )

            if i % 100 == 0:

                summary_str = session.run(merged)
                summary_writer.add_summary(summary_str, i)

            if i % 1000 == 0 or (i + 1) == training_steps:

                checkpoint_path = "%s/checkpoints.ckpt" % exp_path
                saver.save(session, checkpoint_path, global_step=i)
