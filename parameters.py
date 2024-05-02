"""
Functions for parsing command line arguments for different scripts.
"""

import argparse


def parse_arguments():
    """Parse command-line arguments for train.py"""
    parser = argparse.ArgumentParser()

    # paths ---------------------------------------------------------------------------------------
    parser.add_argument("-od", "--output_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="path to directory, where results should be stored")

    parser.add_argument("-dd", "--dataset_dir",
                        type=str,
                        default="/datasets_nas/jowa3080/market_standardized",
                        help="path to directory, where the training data is stored")

    parser.add_argument("-tdd", "--test_data_dir",
                        type=str,
                        default="/datasets_nas/jowa3080/market1501",
                        help="path to directory, where the training data is stored")

    parser.add_argument("-vdd", "--validation_data_dir",
                        type=str,
                        default=None,
                        help="optional path to directory, where the validation data is stored")

    parser.add_argument("-pw", "--pretrain_weights",
                        type=str,
                        default=None,
                        help="path to weights, if model should use pretrained weights")

    # data ----------------------------------------------------------------------------------------
    parser.add_argument("-is", "--input_size",
                        type=int,
                        nargs=2,
                        default=[256, 128],
                        help="which height and width should be used for inputs")

    parser.add_argument("-s", "--sampler",
                        type=str,
                        default="triplet",
                        choices=["triplet", "random", "balanced_triplet"],
                        help="sampling strategy, which should be used")

    parser.add_argument("-bs", "--batch_size",
                        type=int,
                        default=64,
                        help="batch size to use")

    parser.add_argument("-ni", "--number_instances",
                        type=int,
                        default=4,
                        help="number of images per identity per batch")

    parser.add_argument("-rc", "--random_cropping",
                        action="store_false",
                        help="whether random cropping should be disabled")

    parser.add_argument("-rea", "--random_erasing_augmentation",
                        action="store_false",
                        help="whether random erasing augmentation should be disabled")

    # architecture --------------------------------------------------------------------------------
    parser.add_argument("-a", "--architecture",
                        default="baseline",
                        choices=["baseline", "embedding", "mgn"],
                        help="which architecture should be used")

    parser.add_argument("-rv", "--resnet_version",
                        default="pytorch",
                        choices=["pytorch", "keras"],
                        help="whether the pytorch or the keras version of a pretrained ResNet50 should be used. "
                             "For the pytorch version inputs were normalized, which leads to better results.")

    parser.add_argument("-gp", "--global_pooling",
                        type=str,
                        default="average",
                        choices=["average", "max"],
                        help="which pooling operation should be used for aggregation of backbone features")

    parser.add_argument("-ed", "--embedding_dimension",
                        type=int,
                        default=2048,
                        help="dimension of embedding layer, if embedding model is activated")

    parser.add_argument("-lst", "--last_stride",
                        action="store_false",
                        help="whether last stride should be disabled (leads to performance gains for baseline"
                             "architecture, but nor for mgn")

    # loss functions -----------------------------------------------------------------------------
    parser.add_argument("-cl", "--classification_loss",
                        type=str,
                        default="softmax",
                        choices=["softmax", "aaml", "circle", "sphereface", "cosface", "none"],
                        help="which classification lossfunction should be used")

    parser.add_argument("-ml", "--metric_loss",
                        type=str,
                        default="triplet",
                        choices=["triplet", "circle", "none"],
                        help="which metric lossfunction should be used")

    parser.add_argument("-fcl", "--feature_constraint_loss",
                        type=str,
                        default="none",
                        choices=["center", "ring", "none"],
                        help="which feature constraint loss should be used")

    parser.add_argument("-clw", "--classification_loss_weight",
                        type=float,
                        default=1,
                        help="which weight for classification loss should be used")

    parser.add_argument("-mlw", "--metric_loss_weight",
                        type=float,
                        default=1,
                        help="which weight for metric loss should be used")

    parser.add_argument("-fclw", "--feature_constraint_loss_weight",
                        type=float,
                        default=0.0005,
                        help="which weight for feature constraint loss (Ring Loss/Center Loss) should be used")

    parser.add_argument("-cls", "--classification_loss_scale",
                        type=int,
                        default=128,
                        help="value for scale parameter in advanced classification loss functions")

    parser.add_argument("-clm", "--classification_loss_margin",
                        type=float,
                        default=0.25,
                        help="value for margin parameter in advanced classification loss functions")

    parser.add_argument("-tmt", "--triplet_margin_type",
                        type=str,
                        default="hard",
                        choices=["hard", "soft"],
                        help="whether hard or soft margin should be used for triplet loss")

    parser.add_argument("-tmv", "--triplet_margin_value",
                        type=float,
                        default=0.3,
                        help="which value should be used for triplet hard margin")

    parser.add_argument("-mls", "--metric_loss_scale",
                        type=int,
                        default=128,
                        help="value for scale parameter in pairwise circle loss functions")

    parser.add_argument("-mlm", "--metric_loss_margin",
                        type=float,
                        default=0.25,
                        help="value for margin parameter in pairwise circle loss functions")

    parser.add_argument("-wd", "--weight_decay",
                        type=float,
                        default=0.0005,
                        help="factor to use for regularization loss")

    parser.add_argument("-lsm", "--label_smoothing",
                        action="store_false",
                        help="whether label smoothing should be disabled for softmax loss")

    # training hyperparameters ---------------------------------------------------------------------------------
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=120,
                        help="number of epochs to train")

    parser.add_argument("-o", "--optimizer",
                        type=str,
                        default="adam",
                        choices=["adam", "sgd"],
                        help="which optimizer to use")

    parser.add_argument("-slr", "--start_learning_rate",
                        type=float,
                        default=3.5e-4,
                        help="initial learning rate to start training with (after warmup)")

    parser.add_argument("-lrs", "--learning_rate_steps",
                        type=int,
                        nargs="*",
                        default=[40, 70],
                        help="which height and width should be used for inputs")

    parser.add_argument("-fe", "--freeze_epochs",
                        type=int,
                        default=0,
                        help="number of epochs the backbone should be freezed for pretraining")

    parser.add_argument("-we", "--warmup_epochs",
                        type=int,
                        default=10,
                        help="number of epochs for warmup of the learning rate")

    parser.add_argument("-lru", "--learning_rate_updates",
                        type=str,
                        default="epochwise",
                        choices=["epochwise", "batchwise"],
                        help="number of epochs for warmup of the learning rate")

    # inference settings --------------------------------------------------------------------------
    parser.add_argument("-ep", "--evaluation_period",
                        type=int,
                        default=1,
                        help="after how many epochs the model should be evaluated")

    parser.add_argument("-cp", "--checkpoint_period",
                        type=int,
                        default=1,
                        help="after how many epochs checkpoints of the model weights should be saved")

    parser.add_argument("-dm", "--distance_metric",
                        type=str,
                        default="cosine",
                        choices=["cosine", "euclidean"],
                        help="distance metric to use for inference")

    parser.add_argument("-fv", "--feature_vector",
                        type=str,
                        default="after",
                        choices=["after", "before"],
                        help="feature vector to use for inference (before or after BNNeck)")

    parser.add_argument("-tta", "--test_time_augmentation",
                        action="store_true",
                        help="whether to use test time augmentation")

    # other ---------------------------------------------------------------------------------------
    parser.add_argument("-rs", "--random_seed",
                        type=int,
                        default=0,
                        help="random seed to use")

    parser.add_argument("-gc", "--gradient_checkpointing",
                        action="store_false",
                        help="whether gradient checkpointing should be disabled")

    return parser.parse_args()


def parse_configuration():
    """Parse command-line arguments for test_configuration.py"""
    parser = argparse.ArgumentParser()

    # paths ---------------------------------------------------------------------------------------
    parser.add_argument("-c", "--configuration",
                        type=str,
                        default=None,
                        required=True,
                        help="path to directory (not a specific weight file!), where the configuration is stored")
    parser.add_argument("-dm", "--distance_metric",
                        type=str,
                        default="cosine",
                        choices=["cosine", "euclidean"],
                        help="distance metric to use for inference")
    parser.add_argument("-tta", "--test_time_augmentation",
                        action="store_true",
                        help="whether to use test time augmentation")
    parser.add_argument("-tdd", "--test_data_dir",
                        type=str,
                        default="/datasets_nas/jowa3080/market1501",
                        help="path to directory, where the training data is stored")
    parser.add_argument("-fn", "--file_name",
                        type=str,
                        required=True,
                        help="name of the file, where results should be stored (.csv-ending should be included)")

    return parser.parse_args()


def parse_checkpoint():
    """Parse command-line arguments for test_checkpoint.py"""
    parser = argparse.ArgumentParser()

    # paths ---------------------------------------------------------------------------------------
    parser.add_argument("-c", "--checkpoint",
                        type=str,
                        default=None,
                        required=True,
                        help="path to specific weight file of checkpoint")
    parser.add_argument("-dm", "--distance_metric",
                        type=str,
                        default="cosine",
                        choices=["cosine", "euclidean"],
                        help="distance metric to use for inference")
    parser.add_argument("-tta", "--test_time_augmentation",
                        action="store_true",
                        help="whether to use test time augmentation")
    parser.add_argument("-tdd", "--test_data_dir",
                        type=str,
                        default="/datasets_nas/jowa3080/market1501",
                        help="path to directory, where the training data is stored")
    return parser.parse_args()


def parse_datasets():
    """Parse command-line arguments for create_datasets.py"""
    parser = argparse.ArgumentParser()

    # paths ---------------------------------------------------------------------------------------
    parser.add_argument("-id", "--input_dirs",
                        type=str,
                        nargs="*",
                        required=True,
                        help="path to directories, where the datasets are stored")

    parser.add_argument("-dn", "--dataset_names",
                        type=str,
                        nargs="*",
                        required=True,
                        choices=['market', 'duke', 'cuhk', 'msmt'],
                        help="names of the datasets that were given to the parameter --input_dirs"
                             "(they have to be in the same order)")

    parser.add_argument("-od", "--output_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="path to directory, where the standardized Market1501 dataset should be stored stored")

    parser.add_argument("-vd", "--validation_dir",
                        type=str,
                        default=None,
                        help="path to directory, where the validation data should be stored stored (optional)")

    parser.add_argument("-sr", "--split_ratio",
                        type=float,
                        default=0.1,
                        help="ratio of training data that should be separated for validation")

    return parser.parse_args()
