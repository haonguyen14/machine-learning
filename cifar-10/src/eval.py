import tensorflow as tf
import numpy as np
import time
import math
import sys
from datetime import datetime


import input_pipeline as ip
import model as m
from configuration import Configuration


def evaluate(saver, exp_name, top_k):

    with tf.Session() as session:

        ckpt = tf.train.get_checkpoint_state("experiments/%s" % exp_name)

        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
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

                predictions = session.run([top_k])
                true_count += np.sum(predictions)
                step += 1

            precision = float(true_count) / total_sample_count
            print("%s: precision @ %s = %.3f" % (datetime.now(), global_step, precision))
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

if __name__ == "__main__":

    file_name = ["data/test_batch.bin"]
    exp_name = sys.argv[1]
    a_function = tf.nn.relu if sys.argv[2] == "relu" else tf.sigmoid

    pipeline = ip.DataPipeline(file_name)
    test_x, test_y = pipeline.get_batch_op()

    config = Configuration(
        input_size=3072,
        output_size=10,
        batch_size=100,
        num_epoch=1000000,
        a_function=a_function
    )

    convo_model = m.ConvoModel(config)
    convo_model.initialize(test_x)

    predictions = convo_model.infer()

    top_k = tf.nn.in_top_k(predictions, test_y, 1)

    saver = tf.train.Saver(tf.all_variables())

    while True:

        evaluate(saver, exp_name, top_k)
        time.sleep(45)
