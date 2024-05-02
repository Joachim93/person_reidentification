"""
Script for evaluating a single model checkpoint
"""
import json
import argparse
from model.modeling import build_model
from evaluation import inference
from data import create_datasets
import os
from parameters import parse_checkpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

config = parse_checkpoint()
weights = config.checkpoint

# load settings from configuration
base_path = "/".join(weights.split("/")[:-2])
with open(base_path + "/settings.json", "r") as file:
    settings_dict = json.load(file)

# overwrite parameters for inference with arguments of test_checkpoint.py
args = argparse.Namespace(**settings_dict)
args.pretrain_weights = config.checkpoint
args.distance_metric = config.distance_metric
args.test_time_augmentation = config.test_time_augmentation
args.test_data_dir = config.test_data_dir

query_dir = os.path.join(args.test_data_dir, "query")
gallery_dir = os.path.join(args.test_data_dir, "bounding_box_test")

model = build_model(args, training=False)

use_cosine = args.distance_metric == "cosine"

preprocess_func = \
    create_datasets.get_preprocess_func(args,
                                        tuple(args.input_size),
                                        'resize_no_warp_keras_resnet50',
                                        1.0)

inference.evaluate_like_market1501(model, args, query_dir, gallery_dir, preprocess_func,
                                   batch_size=16, cosine=use_cosine, epoch=0)
