"""
Functions for building different network architectures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import os

from losses.center_loss import CenterLoss
from losses.ring_loss import RingLoss
from losses.circle_loss import CircleLoss
from losses.sphereface import Sphereface
from losses.cosface import Cosface
from losses.aaml import AAML

from data import create_datasets, create_mini_batches
import h5py

from model.resnet_pytorch import ResNetPytorch
from model.resnet_keras import ResNetKeras


def build_model(arguments, training=True):
    """
    Function for building different model architectures with or without pretrained weights.

    Parameters
    ----------
    arguments : Namespace
        Contains all arguments, which where defined from command line when running the script.
    training: bool
        Specifies different behavior in training and testing phase

    Returns
    -------
    keras model
    """

    # get layer for classification loss
    if arguments.classification_loss == "aaml":
        cls_layer = AAML
    elif arguments.classification_loss == "circle":
        cls_layer = CircleLoss
    elif arguments.classification_loss == "sphereface":
        cls_layer = Sphereface
    elif arguments.classification_loss == "cosface":
        cls_layer = Cosface
    else:
        cls_layer = layers.Dense

    if arguments.architecture == "mgn":
        model = build_model_mgn(arguments, cls_layer)
    elif arguments.architecture == "embedding":
        model = build_model_embedding(arguments, cls_layer)
    else:
        model = build_model_baseline(arguments, cls_layer)
    if arguments.pretrain_weights:
        query_dir = os.path.join(arguments.test_data_dir, "query")
        load_model(model, arguments, query_dir, training)
    return model


def build_model_baseline(arguments, cls_layer):
    """
    Builds a baseline architecture similar to the one used in Bag of Tricks (https://arxiv.org/abs/1903.07071),
    which can be specified by certain arguments.

    Parameters
    ----------
    arguments : Namespace
        Contains all arguments, which where defined from command line when running the script.
        Used parameter from arguments in this function: [input_size, resnet_version, last_stride,
        classification_loss, feature_constraint_loss, weight_decay, classification_loss_scale,
        classification_loss_margin]
    cls_layer: keras.layers.Layer
        - keras layer, which should be used for the final layer of the model (all custom loss functions
        are implemented as keras layers

    Returns
    -------
    keras model
    """

    input_size = tuple(arguments.input_size)
    input_shape = input_size + (3,)

    if arguments.resnet_version == "pytorch":
        ResNet = ResNetPytorch
    else:
        ResNet = ResNetKeras

    if arguments.last_stride:
        stride = 1
    else:
        stride = 2

    inputs = keras.Input(shape=2048, name="feature_triplet")
    labels = keras.Input(shape=arguments.num_class, name="labels")

    features = layers.BatchNormalization(center=False, gamma_regularizer=l2(arguments.weight_decay),
                                         epsilon=0.00001, momentum=0.9, name="feature_softmax")(inputs)

    if arguments.classification_loss in ("softmax", None):
        logits = cls_layer(arguments.num_class, activation='softmax',
                           kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
                           kernel_regularizer=l2(arguments.weight_decay),
                           use_bias=False, name="logits")(features)
    else:
        logits = cls_layer(scale=arguments.classification_loss_scale, margin=arguments.classification_loss_margin,
                           kernel_regularizer=l2(arguments.weight_decay), name="logits")([features, labels])

    if arguments.feature_constraint_loss == "center":
        centers = CenterLoss(num_classes=arguments.num_class)([inputs, labels])
        output_model = keras.Model([inputs, labels], [logits, inputs, features, centers])
    elif arguments.feature_constraint_loss == "ring":
        ring_loss_layer = RingLoss(radius=1.0,
                                                  loss_type='squared',
                                                  trainable=True,
                                                  name='ring_loss')(features)
        output_model = keras.Model([inputs, labels], [logits, inputs, features, ring_loss_layer])
    else:
        output_model = keras.Model([inputs, labels], [logits, inputs, features])

    model = ResNet(arguments, input_shape, output_model, arguments.weight_decay, stride)

    return model


def build_model_embedding(arguments, cls_layer):
    """
    Builds another baseline architecture similar, which can be specified by certain arguments.
    In contrast to the architecture from build_model_baseline() there is an additional bottleneck layer
    between backbone and final layer instead of a batch normalization neck.

    Parameters
    ----------
    arguments : Namespace
        Contains all arguments, which where defined from command line when running the script.
        Used parameter from arguments in this function: [input_size, resnet_version, last_stride,
        classification_loss, feature_constraint_loss, weight_decay, classification_loss_scale,
        classification_loss_margin, embedding_dimension]
    cls_layer: keras.layers.Layer
        - keras layer, which should be used for the final layer of the model (all custom loss functions
        are implemented as keras layers

    Returns
    -------
    keras model
    """

    input_size = tuple(arguments.input_size)
    input_shape = input_size + (3,)
    if arguments.resnet_version == "pytorch":
        ResNet = ResNetPytorch
    else:
        ResNet = ResNetKeras
    if arguments.last_stride:
        stride = 1
    else:
        stride = 2

    inputs = keras.Input(shape=2048, name="feature_triplet")
    labels = keras.Input(shape=arguments.num_class, name="labels")

    embedding = layers.Dense(arguments.embedding_dimension, kernel_regularizer=l2(arguments.weight_decay),
                             bias_regularizer=l2(arguments.weight_decay))(inputs)
    # embedding = layers.BatchNormalization(center=False, gamma_regularizer=l2(arguments.weight_decay),
    #                                      epsilon=0.00001, momentum=0.9, name="feature_softmax")(embedding)

    if arguments.classification_loss in ("softmax", None):
        logits = cls_layer(arguments.num_class, activation='softmax', name="logits", use_bias=False)(embedding)
        # logits = cls_layer(arguments.num_class, activation='softmax', name="logits")(embedding)
    else:
        logits = cls_layer(scale=arguments.classification_loss_scale, margin=arguments.classification_loss_margin,
                           kernel_regularizer=l2(arguments.weight_decay), name="logits")([embedding, labels])

    if arguments.feature_constraint_loss == "center":
        centers = CenterLoss(num_classes=arguments.num_class, feat_dim=arguments.embedding_dimension)(
            [embedding, labels])
        output_model = keras.Model([inputs, labels], [logits, centers, embedding])
    elif arguments.feature_constraint_loss == "ring":
        ring_loss_layer = RingLoss(radius=1.0,
                                      loss_type='squared',
                                      trainable=True,
                                      name='ring_loss')(embedding)
        output_model = keras.Model([inputs, labels], [logits, ring_loss_layer, embedding])
    else:
        output_model = keras.Model([inputs, labels], [logits, embedding])

    model = ResNet(arguments, input_shape, output_model, arguments.weight_decay, stride=stride)

    return model


def build_model_mgn(arguments, cls_layer):
    """
    Builds an Multiple Granularitiy Network similar to the one used in this paper (https://arxiv.org/abs/1804.01438),
    which can be specified by certain arguments. The idea of this architecture is to extract local features from
    different regions of the input image.

    Parameters
    ----------
    arguments : Namespace
        Contains all arguments, which where defined from command line when running the script.
        Used parameter from arguments in this function: [input_size, resnet_version, last_stride,
        classification_loss, feature_constraint_loss, weight_decay, classification_loss_scale,
        classification_loss_margin]
    cls_layer: keras.layers.Layer
        - keras layer, which should be used for the final layer of the model (all custom loss functions
        are implemented as keras layers

    Returns
    -------
    keras model
    """

    input_size = tuple(arguments.input_size)
    input_shape = input_size + (3,)
    if arguments.resnet_version == "pytorch":
        ResNet = ResNetPytorch
    else:
        ResNet = ResNetKeras
    if arguments.last_stride:
        stride = 1
    else:
        stride = 2

    inputs = keras.Input(shape=(8, 2048), name="feature_triplet")
    labels = keras.Input(shape=arguments.num_class, name="labels")

    features = [tf.squeeze(x, axis=1) for x in tf.split(inputs, 8, axis=1)]
    reduced_features = []
    bn_features = []
    logits = []
    centers = []
    ring_loss_layers = []

    for index, feature in enumerate(features):
        reduced_feature = layers.Dense(256, kernel_regularizer=l2(arguments.weight_decay),
                                       use_bias=False, kernel_initializer="he_uniform",
                                       name=f"embedding_layer_{index}")(feature)

        bn_feature = layers.BatchNormalization(center=False, gamma_regularizer=l2(arguments.weight_decay),
                                               epsilon=0.00001, momentum=0.9)(reduced_feature)

        if arguments.classification_loss in ("softmax", None):
            logit = cls_layer(arguments.num_class, activation='softmax',
                              kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
                              kernel_regularizer=l2(arguments.weight_decay),
                              use_bias=False, name=f"logits_{index}")(bn_feature)
        else:
            logit = cls_layer(scale=arguments.classification_loss_scale, margin=arguments.classification_loss_margin,
                              kernel_regularizer=l2(arguments.weight_decay), name=f"logits_{index}")([bn_feature, labels])

        if arguments.feature_constraint_loss == "center":
            center = CenterLoss(num_classes=arguments.num_class, feat_dim=256)([reduced_feature, labels])
            centers.append(center)
        elif arguments.feature_constraint_loss == "ring":
            ring_loss_layer = RingLoss(radius=1.0,
                                          loss_type='squared',
                                          trainable=True)(bn_feature)
            ring_loss_layers.append(ring_loss_layer)

        logits.append(logit)
        reduced_features.append(reduced_feature)
        bn_features.append(bn_feature)

    outputs = logits + features + reduced_features + bn_features + centers + ring_loss_layers
    outputs.append(tf.concat(bn_features, axis=1))

    output_model = keras.Model([inputs, labels], outputs)

    model = ResNet(arguments, input_shape, output_model, arguments.weight_decay, stride, mgn=True)

    return model


def load_model(model, arguments, query_dir, training):
    """
    Function for loading pretrained weights into the architecture.

    Parameters
    ----------
    model: tf.keras.Model
        keras model in which the pretrained weights should be loaded
    arguments : Namespace
        Contains all arguments, which where defined from command line when running the script.
        Used parameter from arguments in this function: [input_size, num_class, classification_loss,
        architecture, pretrain_weights]
    query_dir: str
        Path to query data from used dataset (used to generate some dummy data, because the model isn't build yet)
    training: bool
        Specifies different behavior in training and testing phase

    Returns
    -------
    keras model

    Notes:
        The current implementation only supports loading pretrained weights into an architecture with the same
        structure. Because feature constraint losses (Ring Loss, Center Loss) create an additional layer with
        weights, they must be integrated or ommited in both models (pretrained and finetuned). If they shouldn't
        contribute to the loss calculation, the argument feature_constraint_loss_weight can be set to 0.
    """
    # Model isn't build before it has seen some inputs.
    # Therefore a dummy input is fed into the model, before loading weights is possible.
    preprocess_func = \
        create_datasets.get_preprocess_func(arguments,
                                            tuple(arguments.input_size),
                                            'resize_no_warp_keras_resnet50',
                                            1.0)
    dummy_sequence = create_mini_batches.ValidationDataSequence(query_dir,
                                                                batch_size=16,
                                                                num_class=arguments.num_class,
                                                                preprocess=preprocess_func,
                                                                test_mode=True)
    dummy_input = dummy_sequence[0]
    model(dummy_input, training=True, mgn=bool(arguments.architecture == "mgn"), checkpointing=True)
    # if Circle Loss/AAML are initialized with Softmax weights, there will be a shape missmatch
    model.load_weights(arguments.pretrain_weights, by_name=True, skip_mismatch=True)

    if training:
        # pretrained weight of Softmax layer must be transposed for all advanced softmax variants
        if arguments.classification_loss in ["aaml", "circle", "sphereface", "cosface"]:
            weights_file = h5py.File(arguments.pretrain_weights, "r")
            if arguments.architecture == "mgn":
                layers = ["logits_0", "logits_1", "logits_2", "logits_3", "logits_4", "logits_5", "logits_6", "logits_7"]

                # old models
                # old_layers = ["dense_1", "dense_3", "dense_5", "dense_7", "dense_9", "dense_11", "dense_13", "dense_15"]
                # for layer, old_layer in zip(layers, old_layers):
                #     aaml_weights = weights_file["functional_1"][old_layer]["kernel:0"][()].transpose()
                #     model.layers[-1].get_layer(layer).set_weights([aaml_weights])
                #     print("loaded")

                for layer in layers:
                    classif_weights = weights_file["functional_1"][layer]["kernel:0"][()].transpose()
                    model.layers[-1].get_layer(layer).set_weights([classif_weights])
                    print("loaded")

            else:
                classif_weights = weights_file["functional_1"]["logits"]["kernel:0"][()].transpose()
                # classif_weights = weights_file["functional_1"]["logits_out"]["kernel:0"][()].transpose()
                model.layers[-1].get_layer("logits").set_weights([classif_weights])
                print("loaded")

    return model
