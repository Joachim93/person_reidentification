#!/usr/bin/env python
# We want one GPU for this script
#SBATCH --gres=gpu:1

# We want to submit this job to the partition named 'long'.
#SBATCH -p long

# We will need two hours of compute time
# IMPORTANT: If you exceed this timelimit your job will NOT be canceled. BUT for
# SLURM to be able to schedule efficiently a reasonable estimate is needed.
#SBATCH -t 20:00:00

# We want to be notified by email when the jobs starts / ends
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=joachim.wagner@tu-ilmenau.de


"""
Main Script for training models
"""

import sys
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy.random import seed
import random
import tensorflow.keras.backend as K
import json
import shutil
from collections import defaultdict
from data.create_datasets import get_preprocess_func
from data.create_mini_batches import get_data_sampler
from training.lr_scheduler import get_lr_scheduler
from parameters import parse_arguments
from model.modeling import build_model
from training.logger import CSVLogger
from evaluation import validation, inference
from losses.triplet_loss import TripletLoss
from losses.pairwise_circle import PairwiseCircleLoss

sys.path.append(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

data_format = 'channels_last'
K.set_image_data_format(data_format)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parse_arguments()

# convert string values to None type
if args.classification_loss == "none":
    args.classification_loss = None
if args.metric_loss == "none":
    args.metric_loss = None
if args.feature_constraint_loss == "none":
    args.feature_constraint_loss = None


# create directories and save settings
full_path = os.path.join(args.output_dir, str(args.random_seed))

# existing directory will be overwritten
if os.path.exists(full_path):
    shutil.rmtree(full_path)
os.makedirs(os.path.join(full_path, "weights"))

# set random seeds for determinism
random.seed(args.random_seed)
seed(args.random_seed)
tf.random.set_seed(args.random_seed)
os.environ['PYTHONHASHSEED'] = str(args.random_seed)

scheduler = get_lr_scheduler(args)

input_size = tuple(args.input_size)
input_shape = input_size + (3,)
preprocess_func = get_preprocess_func(args,
                                      input_size,
                                      'resize_no_warp_keras_resnet50',
                                      1.0)

query_dir = os.path.join(args.test_data_dir, "query")
gallery_dir = os.path.join(args.test_data_dir, "bounding_box_test")

feature = 2 if args.feature_vector == "after" else 1
use_cosine = args.distance_metric == "cosine"

if args.validation_data_dir:
    train_dat_seq, val_dat_seq = get_data_sampler(args, preprocess_func)
else:
    train_dat_seq = get_data_sampler(args, preprocess_func)


# Output from AAML and Circle Loss are Logits and not probabilities
if args.classification_loss in ["aaml", "circle", "sphereface", "cosface"]:
    softmax_loss = keras.losses.CategoricalCrossentropy(from_logits=True)
else:
    if args.label_smoothing:
        softmax_loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    else:
        softmax_loss = keras.losses.CategoricalCrossentropy()


if args.metric_loss == "circle":
    metric_loss = PairwiseCircleLoss(args.metric_loss_scale, args.metric_loss_margin)
else:
    metric_loss = TripletLoss(args.triplet_margin_type, args.triplet_margin_value)

if args.optimizer == "adam":
    optimizer = keras.optimizers.Adam(epsilon=1e-8)
else:
    optimizer = keras.optimizers.SGD(momentum=0.9, decay=0.0005)

optimizer_center = keras.optimizers.SGD(learning_rate=0.5)
args.num_class = train_dat_seq.num_class
model = build_model(args)

# save settings
with open(os.path.join(full_path, "settings.json"), "w") as file:
    json.dump(vars(args), file)


def step_train(images, labels, train_vars):

    with tf.GradientTape() as tape:

        if args.architecture == "mgn":
            outputs = model([images, labels], training=True, checkpointing=args.gradient_checkpointing, mgn=True)

            if args.classification_loss:
                loss_softmax = 0
                for logit in outputs[:8]:
                    loss_softmax += 1/8 * softmax_loss(labels[0], logit)
                loss_softmax = args.classification_loss_weight * loss_softmax
            else:
                loss_softmax = tf.constant(0, dtype=tf.float32)
            if args.metric_loss:
                if args.classification_loss:
                    loss_metric = 1 / 8 * metric_loss.compute_loss(np.argmax(labels[0], axis=1), outputs[16]) + \
                                   1 / 8 * metric_loss.compute_loss(np.argmax(labels[0], axis=1), outputs[17]) + \
                                   1 / 8 * metric_loss.compute_loss(np.argmax(labels[0], axis=1), outputs[20])
                else:
                    loss_metric = 0
                    for feature in outputs[24:32]:
                        loss_metric += 1 / 8 * metric_loss.compute_loss(np.argmax(labels[0], axis=1), feature)
                loss_metric = args.metric_loss_weight * loss_metric
            else:
                loss_metric = tf.constant(0, dtype=tf.float32)
            if args.feature_constraint_loss:
                loss_constraint = 0
                for constraint in outputs[32:40]:
                    loss_constraint += 1 / 8 * constraint
                loss_constraint = args.feature_constraint_loss_weight * loss_constraint
            else:
                loss_constraint = tf.constant(0, dtype=tf.float32)
            loss_reg = 0.5 * sum(model.losses)
            loss = loss_softmax + loss_metric + loss_constraint + loss_reg

        elif args.architecture == "embedding":
            if args.feature_constraint_loss:
                logits, loss_constraint, features = model([images, labels], training=True,
                                                     checkpointing=args.gradient_checkpointing)
            else:
                logits, features = model([images, labels], training=True, checkpointing=args.gradient_checkpointing)

            if args.classification_loss:
                loss_softmax = args.classification_loss_weight * softmax_loss(labels[0], logits)
            else:
                loss_softmax = tf.constant(0, dtype=tf.float32)
            if args.metric_loss:
                loss_metric = args.metric_loss_weight * metric_loss.compute_loss(np.argmax(labels[0], axis=1), features)
            else:
                loss_metric = tf.constant(0, dtype=tf.float32)
            if args.feature_constraint_loss:
                loss_constraint = args.feature_constraint_loss_weight * loss_constraint
            else:
                loss_constraint = tf.constant(0, dtype=tf.float32)
            loss_reg = 0.5 * sum(model.losses)
            loss = loss_softmax + loss_metric + loss_constraint + loss_reg

        else:
            if args.feature_constraint_loss:
                logits, features, features2, loss_constraint = model([images, labels], training=True,
                                                         checkpointing=args.gradient_checkpointing)
            else:
                logits, features, features2 = model([images, labels], training=True,
                                                    checkpointing=args.gradient_checkpointing)

            if args.classification_loss:
                loss_softmax = args.classification_loss_weight * softmax_loss(labels[0], logits)
            else:
                loss_softmax = tf.constant(0, dtype=tf.float32)
            if args.metric_loss:
                if args.classification_loss:
                    loss_metric = args.metric_loss_weight * metric_loss.compute_loss(np.argmax(labels[0], axis=1), features)
                else:
                    loss_metric = args.metric_loss_weight * metric_loss.compute_loss(np.argmax(labels[0], axis=1), features2)
            else:
                loss_metric = tf.constant(0, dtype=tf.float32)
            if args.feature_constraint_loss:
                loss_constraint = args.feature_constraint_loss_weight * loss_constraint
            else:
                loss_constraint = tf.constant(0, dtype=tf.float32)
            loss_reg = 0.5 * sum(model.losses)
            loss = loss_softmax + loss_metric + loss_constraint + loss_reg


    grads = tape.gradient(loss, train_vars)

    if args.feature_constraint_loss == "center":

        if args.architecture == "mgn":
            optimizer.apply_gradients(zip(grads[:-8], train_vars[:-8]))
            # if center loss was used for the pretrained weights, but should not used for fine tuning
            if args.feature_constraint_loss_weight != 0:
                new_grads = [grad / args.feature_constraint_loss_weight for grad in grads[-8:]]
                optimizer_center.apply_gradients(
                    zip(new_grads, train_vars[-8:]))
        else:
            optimizer.apply_gradients(zip(grads[:-1], train_vars[:-1]))
            # possible use case: if center loss was used for the pretraining, but should not be used for fine tuning
            if args.feature_constraint_loss_weight != 0:
                optimizer_center.apply_gradients(
                    zip([grads[-1] / args.feature_constraint_loss_weight], [train_vars[-1]]))

    else:
        optimizer.apply_gradients(zip(grads, train_vars))

    return loss, loss_softmax, loss_metric, loss_constraint, loss_reg


logger = CSVLogger(os.path.join(full_path, 'training.csv'))
# logger_market = CSVLogger(os.path.join(full_path, 'market.csv'))

for epoch in range(1, args.epochs+1):
    train_dat_seq.create_batches()

    # freeze backbone at begin of training
    if epoch <= args.freeze_epochs:
        for layer in model.layers[:-1]:
            layer.trainable = False
        print("backbone freezed")
    elif epoch == (args.freeze_epochs + 1):
        for layer in model.layers[:-1]:
            layer.trainable = True
        if args.optimizer == "adam":
            optimizer = keras.optimizers.Adam(epsilon=1e-8)
        else:
            optimizer = keras.optimizers.SGD(momentum=0.9, decay=0.0005)

    if args.learning_rate_updates == "epochwise":
        optimizer.lr = scheduler.get_lr(epoch-1)

    trainable_vars = model.trainable_variables

    print("learning_rate:{}".format(optimizer.lr.numpy()))
    print("Epoch {}/{}".format(epoch, args.epochs))

    bar = keras.utils.Progbar(len(train_dat_seq), unit_name="sample")
    log_values = []
    for batch, (X, y) in enumerate(train_dat_seq):

        if args.learning_rate_updates == "batchwise":
            optimizer.lr = scheduler.get_lr(epoch, batch, len(train_dat_seq), args.epochs)

        loss_out, loss_softmax, loss_metric, loss_constraint, loss_reg = step_train(X[0], y, trainable_vars)
        log_values.append(("loss", loss_out.numpy().item()))
        log_values.append(("loss_softmax", loss_softmax.numpy().item()))
        log_values.append(("loss_metric", loss_metric.numpy().item()))
        log_values.append(("loss_constraint", loss_constraint.numpy().item()))
        log_values.append(("loss_reg", loss_reg.numpy().item()))
        bar.add(1, log_values)

    # calculate mean values for logging
    log_dict = defaultdict(float)
    for key, value in log_values:
        log_dict[key] += value

    log_dict = {key: round(value / len(train_dat_seq), 4) for key, value in log_dict.items()}

    if epoch % args.checkpoint_period == 0:
        model.save_weights(os.path.join(full_path, "weights", str(epoch) + ".h5"))

    if epoch % args.evaluation_period == 0:

        if args.validation_data_dir:
            # logs_market = inference.evaluate_like_market1501(model, args, query_dir, gallery_dir, preprocess_func,
            #                                                  batch_size=16, cosine=use_cosine, epoch=epoch)
            # logger_market.write_logs(logs_market)

            logs = validation.on_epoch_end(model, args, val_dat_seq, batch_size=16, epoch=epoch)
            logs.update(log_dict)
            logger.write_logs(logs)
        else:
            logs = inference.evaluate_like_market1501(model, args, query_dir, gallery_dir, preprocess_func,
                                                      batch_size=16, cosine=use_cosine, epoch=epoch)

            logs.update(log_dict)
            logger.write_logs(logs)
