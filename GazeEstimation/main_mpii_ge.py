#!/usr/bin/env python3
"""
Main script for training the DPG model for within-Columbia evaluations.
"""

import argparse
import coloredlogs
import tensorflow as tf
from math import floor
import numpy as np


if __name__ == "__main__":

    # Set global log level
    parser = argparse.ArgumentParser(description="Train the Deep Pictorial Gaze model.")
    parser.add_argument(
        "-v",
        type=str,
        help="logging level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
    )
    parser.add_argument(
        "--train_path",
        type=str,
        help="training data path",
        default="../datasets/columbia.h5",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        help="test data path",
        default="../datasets/columbia.h5",
    )
    parser.add_argument("--oh", type=int, help="image height", default=64)
    parser.add_argument("--ow", type=int, help="image width", default=64)
    parser.add_argument("--ids", type=int, help="id of validation fold")
    parser.add_argument("--log_dir", type=str, default="outputs")
    parser.add_argument(
        "--epoch", type=int, help="number of training epochs", default=20
    )
    parser.add_argument(
        "--n_train", type=int, help="number of participants in training", default=15
    )
    parser.add_argument(
        "--n_test", type=int, help="number of participants in testing", default=56
    )
    args = parser.parse_args()
    coloredlogs.install(
        datefmt="%d/%m %H:%M",
        fmt="%(asctime)s %(levelname)s %(message)s",
        level=args.v.upper(),
    )

    n_folds = 5
    n_people = args.n_train
    n_parts_train = args.n_train
    n_parts_test = args.n_test
    fold_size = floor(n_people / n_folds)
    for fold_id in range(n_folds):
        # for ts in range(0, 10, 2):
        # val_range = range(fold_id, n_people)
        # val_range = range(fold_id*fold_size, (fold_id+1)*fold_size)
        # val_ids = sorted(["%04d" % (j + 1) for j in val_range])
        # train_range = range(fold_id, n_people)
        # train_range = np.random.choice(n_people, n_people-fold_id, replace=False)
        # train_ids = sorted(["%04d" % (j + 1) for j in train_range])
        # train_ids = sorted(["%04d" % (j + 1) for j in range(n_people) if j not in list(val_range)])
        # train_range = ["%04d" % (j + 1) for j in range(n_people) if j not in list(val_range)]
        # train_ids = sorted(np.random.choice(train_range, n_people-ts-fold_size, replace=False))

        val_ids = sorted(["%04d" % (j + 1) for j in range(n_parts_test)])
        train_ids = sorted(["%04d" % (j + 1) for j in range(n_parts_train)])

        # Initialize Tensorflow session
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.ERROR)
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

            # Declare some parameters
            batch_size = 32

            # Define training data source
            from datasources import HDF5Source

            # Define model
            from models import DPG

            model = DPG(
                session,
                learning_schedule=[
                    {
                        "loss_terms_to_optimize": {
                            "combined_loss": ["hourglass", "densenet"],
                        },
                        "metrics": ["gaze_mse", "gaze_ang"],
                        "learning_rate": 0.0002,
                    },
                ],
                extra_tags=[str(fold_id)],
                # extra_tags=[str(fold_id)+'_'+str(n_people-ts-fold_size)],
                # Data sources for training (and testing).
                train_data={
                    "mpi": HDF5Source(
                        session,
                        data_format="NCHW",
                        batch_size=batch_size,
                        keys_to_use=["train/" + s for s in train_ids],
                        hdf_path=args.train_path,
                        eye_image_shape=(args.oh, args.ow),
                        testing=False,
                        min_after_dequeue=30000,
                        staging=True,
                        shuffle=True,
                    ),
                },
                test_data={
                    "mpi": HDF5Source(
                        session,
                        data_format="NCHW",
                        batch_size=batch_size,
                        keys_to_use=["train/" + s for s in val_ids],
                        hdf_path=args.test_path,
                        eye_image_shape=(args.oh, args.ow),
                        testing=True,
                    ),
                },
                log_dir=args.log_dir,
            )

            # Train this model for a set number of epochs
            model.train(
                num_epochs=args.epoch,
            )

            model.__del__()
            session.close()
            del session
