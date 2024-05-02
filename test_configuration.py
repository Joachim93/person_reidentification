"""
Script for evaluating a multiple model checkpoints
"""

import sys
import os
import json
import argparse
from model.modeling import load_model, build_model
from data import create_datasets, create_mini_batches
from training.logger import CSVLogger
from parameters import parse_configuration
from evaluation import inference

sys.path.append(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

config = parse_configuration()
base_path = config.configuration

with open(base_path + "/settings.json", "r") as file:
    settings_dict = json.load(file)

args = argparse.Namespace(**settings_dict)

model = build_model(args, training=False)

logger = CSVLogger(os.path.join(base_path, config.file_name))

args.distance_metric = config.distance_metric
args.test_time_augmentation = config.test_time_augmentation
args.test_data_dir = config.test_data_dir

query_dir = os.path.join(args.test_data_dir, "query")
gallery_dir = os.path.join(args.test_data_dir, "bounding_box_test")

for epoch in range(args.epochs):

    weight_path = "/".join([base_path, "weights", str(epoch+1)+".h5"])
    args.pretrain_weights = weight_path
    load_model(model, args, query_dir, training=False)

    use_cosine = args.distance_metric == "cosine"

    preprocess_func = \
        create_datasets.get_preprocess_func(args,
                                            tuple(args.input_size),
                                            'resize_no_warp_keras_resnet50',
                                            1.0)
    # val_dat_seq = \
    #     create_mini_batches.ValidationDataSequence(args.test_data_dir,
    #                                                args.batch_size,
    #                                                args.num_class,
    #                                                preprocess=preprocess_func,
    #                                                y_true_to_categorical=True)

    logs = inference.evaluate_like_market1501(model, args, query_dir, gallery_dir, preprocess_func,
                                              batch_size=16, cosine=use_cosine, epoch=epoch+1)

    logger.write_logs(logs)
