import tensorflow as tf
import numpy as np
import time
import math
import sys
from datetime import datetime


import input_pipeline as ip
import model as m
from configuration import Configuration


def evaluate(saver, exp_name, top_k, summary_writer, summary_op, dropout, prev_step):

    with tf.Session() as session:

        ckpt = tf.train.get_checkpoint_state("experiments/%s" % exp_name)

        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]

            if prev_step == global_step:
                return

            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint file found")
            return

        coord = tf.train.Coordinator()

        threads = []

        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(
                                session,
                                coord=coord,
                                daemon=True,
                                start=True))

        num_iters = int(math.ceil(10000 / 100))
        true_count = 0
        total_sample_count = num_iters * 100
        step = 0

        try:
            while step < num_iters:

                predictions = session.run([top_k], feed_dict={dropout: 1.0})
                true_count += np.sum(predictions)
                step += 1

            precision = float(true_count) / total_sample_count
            print("%s: precision @ %s = %.3f" % (datetime.now(), global_step, precision))

            summary = tf.Summary()
            summary.ParseFromString(session.run(summary_op, feed_dict={dropout: 1.0}))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

    return global_step

if __name__ == "__main__":

    file_name = ["data/test_batch.bin"]
    exp_name = sys.argv[1]
    a_function = tf.nn.relu if sys.argv[2] == "relu" else tf.sigmoid

    pipeline = ip.DataPipeline(file_name)
    test_x, test_y = pipeline.get_batch_op()

    config = Configuration(
        input_size=3072,
        output_size=10,
        examples_per_epoches=50000,
        batch_size=100,
        a_function=a_function
     )

    dropout = tf.placeholder(tf.float32)

    convo_model = m.ConvoModel(config)
    convo_model.initialize(test_x, dropout)

    predictions = convo_model.infer()

    top_k = tf.nn.in_top_k(predictions, test_y, 1)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter("experiments/%s" % exp_name)

    saver = tf.train.Saver(tf.all_variables())

    prev_step = -1

    while True:

        prev_step = evaluate(saver, exp_name, top_k, summary_writer, summary_op, dropout, prev_step)
        time.sleep(45)
