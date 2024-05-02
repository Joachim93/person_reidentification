"""
Script for Resnet50 architecture with pretrained weights from pytorch
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

pytorch_weights = np.load("/results_nas/jowa3080/resnet_pytorch.npy", allow_pickle=True).item()


def load_pytorch_weights(model, weights_dict):
    """
    Function for loading the ImageNet weights from Pytorch ResNet.

    Parameters
    ----------
    model: keras model
    weights_dict: dictionary

    Returns
    -------
    keras model
    """
    tf_layer_names = [layer.name for layer in model.layers]
    for layer in tf_layer_names:
        if 'conv' in layer:
            tf_conv = model.get_layer(layer)
            tf_conv.set_weights(weights_dict[layer])
        elif 'bn' in layer:
            tf_bn = model.get_layer(layer)
            tf_bn.set_weights(weights_dict[layer])
        elif 'downsample.0' in layer:
            tf_downsample = model.get_layer(layer)
            tf_downsample.set_weights(weights_dict[layer])
        elif 'downsample.1' in layer:
            tf_downsample = model.get_layer(layer)
            tf_downsample.set_weights(weights_dict[layer])
        elif 'fc' in layer:
            tf_fc = model.get_layer(layer)
            tf_fc.set_weights(weights_dict[layer])


def identity_block(input_shape, kernel_size, filters, stage, block, decay):
    """The identity block is the block that has no conv layer at shortcut.

    Arguments:
      input_shape: input shape
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      decay: weight decay factor to use for all layers

    Returns:
      keras model
    """
    filters1, filters2, filters3 = filters

    base_name = "layer" + str(stage) + "." + str(block)

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(decay),
                      name=base_name + '.conv1', use_bias=False)(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=base_name + '.bn1', beta_regularizer=l2(decay), gamma_regularizer=l2(decay), epsilon=0.00001, momentum=0.9)(x)

    x = layers.Activation('relu', name=base_name + '.relu1')(x)

    x = layers.ZeroPadding2D(padding=(1, 1), name='pad1')(x)
    x = layers.Conv2D(
        filters2, kernel_size, kernel_initializer='he_normal', kernel_regularizer=l2(decay),
        padding='valid', name=base_name + '.conv2', use_bias=False)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=base_name + '.bn2', beta_regularizer=l2(decay), gamma_regularizer=l2(decay), epsilon=0.00001, momentum=0.9)(x)
    x = layers.Activation('relu', name=base_name + '.relu2')(x)

    x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(decay),
                      name=base_name + '.conv3', use_bias=False)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=base_name + '.bn3', beta_regularizer=l2(decay), gamma_regularizer=l2(decay), epsilon=0.00001, momentum=0.9)(x)

    x = layers.add([x, inputs], name=base_name + '.add')
    x = layers.Activation('relu', name=base_name + '.relu3')(x)

    model = keras.Model(inputs, x)
    load_pytorch_weights(model, pytorch_weights)

    return model


def conv_block(input_shape, kernel_size, filters, stage, block, decay, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    Arguments:
      input_shape: input shape of images
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      decay: weight decay factor to use for all layers
      strides: Strides for the first conv layer in the block.

    Returns:
      keras model

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    base_name = "layer" + str(stage) + "." + str(block)

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(
        filters1, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(decay),
        name=base_name + '.conv1', use_bias=False)(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=base_name + '.bn1', beta_regularizer=l2(decay), gamma_regularizer=l2(decay), epsilon=0.00001, momentum=0.9)(x)
    x = layers.Activation('relu', name=base_name + '.relu1')(x)

    x = layers.ZeroPadding2D(padding=(1, 1), name='pad1')(x)
    x = layers.Conv2D(
        filters2, kernel_size, padding='valid', strides=strides, kernel_initializer='he_normal', kernel_regularizer=l2(decay),
        name=base_name + '.conv2', use_bias=False)(x)

    x = layers.BatchNormalization(axis=bn_axis, name=base_name + '.bn2', beta_regularizer=l2(decay), gamma_regularizer=l2(decay), epsilon=0.00001, momentum=0.9)(x)
    x = layers.Activation('relu', name=base_name + '.relu2')(x)

    x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(decay),
                      name=base_name + '.conv3', use_bias=False)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=base_name + '.bn3', beta_regularizer=l2(decay), gamma_regularizer=l2(decay), epsilon=0.00001, momentum=0.9)(x)

    shortcut = layers.Conv2D(
        filters3, (1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer=l2(decay),
        name=base_name + '.downsample.0', use_bias=False)(inputs)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=base_name + '.downsample.1', beta_regularizer=l2(decay), gamma_regularizer=l2(decay), epsilon=0.00001, momentum=0.9)(shortcut)

    x = layers.add([x, shortcut], name=base_name + '.add')
    x = layers.Activation('relu', name=base_name + '.relu3')(x)

    model = keras.Model(inputs, x)
    load_pytorch_weights(model, pytorch_weights)

    return model


def first_block(input_shape, decay):
    """
    First block of ResNet50.

    Arguments:
      input_shape: input shape of images
      decay: weight decay factor to use for all layers

    Returns:
      keras model
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name='pad1')(inputs)
    x = layers.Conv2D(
        64, (7, 7), strides=(2, 2), kernel_initializer='he_normal', kernel_regularizer=l2(decay),
        padding='valid', name='conv1', use_bias=False)(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn1', beta_regularizer=l2(decay), gamma_regularizer=l2(decay), epsilon=0.00001, momentum=0.9)(x)
    x = layers.Activation('relu', name="relu")(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pad2')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), name="maxpool")(x)

    model = keras.Model(inputs, x)
    load_pytorch_weights(model, pytorch_weights)

    return model




class ResNetPytorch(Model):
    def __init__(self, args, input_shape, output_layers, decay, stride=2, mgn=False):
        super(ResNetPytorch, self).__init__()
        self.block_1 = first_block(input_shape, decay=decay)
        self.block_2a = conv_block(self.block_1.output_shape[1:], 3, [64, 64, 256], stage=1, block=0, decay=decay, strides=(1, 1))
        self.block_2b = identity_block(self.block_2a.output_shape[1:], 3, [64, 64, 256], stage=1, decay=decay,
                                       block=1)
        self.block_2c = identity_block(self.block_2b.output_shape[1:], 3, [64, 64, 256], stage=1, decay=decay,
                                       block=2)

        self.block_3a = conv_block(self.block_2c.output_shape[1:], 3, [128, 128, 512], stage=2, block=0, decay=decay)
        self.block_3b = identity_block(self.block_3a.output_shape[1:], 3, [128, 128, 512], stage=2, block=1, decay=decay)
        self.block_3c = identity_block(self.block_3b.output_shape[1:], 3, [128, 128, 512], stage=2, block=2, decay=decay)
        self.block_3d = identity_block(self.block_3c.output_shape[1:], 3, [128, 128, 512], stage=2, block=3, decay=decay)

        self.block_4a = conv_block(self.block_3d.output_shape[1:], 3, [256, 256, 1024], stage=3, block=0, decay=decay)
        self.block_4b = identity_block(self.block_4a.output_shape[1:], 3, [256, 256, 1024], stage=3, block=1, decay=decay)
        self.block_4c = identity_block(self.block_4b.output_shape[1:], 3, [256, 256, 1024], stage=3, block=2, decay=decay)
        self.block_4d = identity_block(self.block_4c.output_shape[1:], 3, [256, 256, 1024], stage=3, block=3, decay=decay)
        self.block_4e = identity_block(self.block_4d.output_shape[1:], 3, [256, 256, 1024], stage=3, block=4, decay=decay)
        self.block_4f = identity_block(self.block_4e.output_shape[1:], 3, [256, 256, 1024], stage=3, block=5, decay=decay)

        self.block_5a = conv_block(self.block_4f.output_shape[1:], 3, [512, 512, 2048], stage=4, block=0, decay=decay, strides=stride)
        self.block_5b = identity_block(self.block_5a.output_shape[1:], 3, [512, 512, 2048], stage=4, block=1, decay=decay)
        self.block_5c = identity_block(self.block_5b.output_shape[1:], 3, [512, 512, 2048], stage=4, block=2, decay=decay)

        if mgn:
            self.block_4b_local_2 = identity_block(self.block_4a.output_shape[1:], 3, [256, 256, 1024], stage=3, block=1, decay=decay)
            self.block_4c_local_2 = identity_block(self.block_4b_local_2.output_shape[1:], 3, [256, 256, 1024], stage=3, block=2, decay=decay)
            self.block_4d_local_2 = identity_block(self.block_4c_local_2.output_shape[1:], 3, [256, 256, 1024], stage=3, block=3, decay=decay)
            self.block_4e_local_2 = identity_block(self.block_4d_local_2.output_shape[1:], 3, [256, 256, 1024], stage=3, block=4, decay=decay)
            self.block_4f_local_2 = identity_block(self.block_4e_local_2.output_shape[1:], 3, [256, 256, 1024], stage=3, block=5, decay=decay)

            self.block_5a_local_2 = conv_block(self.block_4f_local_2.output_shape[1:], 3, [512, 512, 2048], stage=4, block=0, decay=decay,
                                       strides=1)
            self.block_5b_local_2 = identity_block(self.block_5a_local_2.output_shape[1:], 3, [512, 512, 2048], stage=4, block=1, decay=decay)
            self.block_5c_local_2 = identity_block(self.block_5b_local_2.output_shape[1:], 3, [512, 512, 2048], stage=4, block=2, decay=decay)

            self.block_4b_local_3 = identity_block(self.block_4a.output_shape[1:], 3, [256, 256, 1024], stage=3,
                                                   block=1, decay=decay)
            self.block_4c_local_3 = identity_block(self.block_4b_local_3.output_shape[1:], 3, [256, 256, 1024], stage=3,
                                                   block=2, decay=decay)
            self.block_4d_local_3 = identity_block(self.block_4c_local_3.output_shape[1:], 3, [256, 256, 1024], stage=3,
                                                   block=3, decay=decay)
            self.block_4e_local_3 = identity_block(self.block_4d_local_3.output_shape[1:], 3, [256, 256, 1024], stage=3,
                                                   block=4, decay=decay)
            self.block_4f_local_3 = identity_block(self.block_4e_local_3.output_shape[1:], 3, [256, 256, 1024], stage=3,
                                                   block=5, decay=decay)

            self.block_5a_local_3 = conv_block(self.block_4f_local_3.output_shape[1:], 3, [512, 512, 2048], stage=4,
                                               block=0, decay=decay,
                                               strides=1)
            self.block_5b_local_3 = identity_block(self.block_5a_local_3.output_shape[1:], 3, [512, 512, 2048], stage=4,
                                                   block=1, decay=decay)
            self.block_5c_local_3 = identity_block(self.block_5b_local_3.output_shape[1:], 3, [512, 512, 2048], stage=4,
                                                   block=2, decay=decay)

        if args.global_pooling == "average":
            self.pool = layers.GlobalAveragePooling2D(data_format=K.image_data_format())
        else:
            self.pool = layers.GlobalMaxPool2D(data_format=K.image_data_format())
        self.last = output_layers

    def call(self, x, training=False, checkpointing=False, mgn=False):

        if checkpointing:
            out = tf.recompute_grad(self.block_1)(x[0], training=training)
            out = tf.recompute_grad(self.block_2a)(out, training=training)
            out = tf.recompute_grad(self.block_2b)(out, training=training)
            out = tf.recompute_grad(self.block_2c)(out, training=training)
            out = tf.recompute_grad(self.block_3a)(out, training=training)
            out = tf.recompute_grad(self.block_3b)(out, training=training)
            out = tf.recompute_grad(self.block_3c)(out, training=training)
            out = tf.recompute_grad(self.block_3d)(out, training=training)
            out = tf.recompute_grad(self.block_4a)(out, training=training)

            if mgn:
                branch_global = tf.recompute_grad(self.block_4b)(out, training=training)
                branch_global = tf.recompute_grad(self.block_4b)(branch_global, training=training)
                branch_global = tf.recompute_grad(self.block_4c)(branch_global, training=training)
                branch_global = tf.recompute_grad(self.block_4d)(branch_global, training=training)
                branch_global = tf.recompute_grad(self.block_4e)(branch_global, training=training)
                branch_global = tf.recompute_grad(self.block_4f)(branch_global, training=training)
                branch_global = tf.recompute_grad(self.block_5a)(branch_global, training=training)
                branch_global = tf.recompute_grad(self.block_5b)(branch_global, training=training)
                branch_global = tf.recompute_grad(self.block_5c)(branch_global, training=training)
                branch_global = self.pool(branch_global, training=training)

                branch_local_2 = tf.recompute_grad(self.block_4b_local_2)(out, training=training)
                branch_local_2 = tf.recompute_grad(self.block_4b_local_2)(branch_local_2, training=training)
                branch_local_2 = tf.recompute_grad(self.block_4c_local_2)(branch_local_2, training=training)
                branch_local_2 = tf.recompute_grad(self.block_4d_local_2)(branch_local_2, training=training)
                branch_local_2 = tf.recompute_grad(self.block_4e_local_2)(branch_local_2, training=training)
                branch_local_2 = tf.recompute_grad(self.block_4f_local_2)(branch_local_2, training=training)
                branch_local_2 = tf.recompute_grad(self.block_5a_local_2)(branch_local_2, training=training)
                branch_local_2 = tf.recompute_grad(self.block_5b_local_2)(branch_local_2, training=training)
                branch_local_2 = tf.recompute_grad(self.block_5c_local_2)(branch_local_2, training=training)
                branch_local_2_part_1 = branch_local_2[:, :branch_local_2.shape[1]//2]
                branch_local_2_part_2 = branch_local_2[:, branch_local_2.shape[1]//2:]

                pool_local_2 = self.pool(branch_local_2, training=training)
                pool_local_2_part_1 = self.pool(branch_local_2_part_1, training=training)
                pool_local_2_part_2 = self.pool(branch_local_2_part_2, training=training)

                branch_local_3 = tf.recompute_grad(self.block_4b_local_3)(out, training=training)
                branch_local_3 = tf.recompute_grad(self.block_4b_local_3)(branch_local_3, training=training)
                branch_local_3 = tf.recompute_grad(self.block_4c_local_3)(branch_local_3, training=training)
                branch_local_3 = tf.recompute_grad(self.block_4d_local_3)(branch_local_3, training=training)
                branch_local_3 = tf.recompute_grad(self.block_4e_local_3)(branch_local_3, training=training)
                branch_local_3 = tf.recompute_grad(self.block_4f_local_3)(branch_local_3, training=training)
                branch_local_3 = tf.recompute_grad(self.block_5a_local_3)(branch_local_3, training=training)
                branch_local_3 = tf.recompute_grad(self.block_5b_local_3)(branch_local_3, training=training)
                branch_local_3 = tf.recompute_grad(self.block_5c_local_3)(branch_local_3, training=training)
                branch_local_3_part_1 = branch_local_3[:, :branch_local_2.shape[1]//3]
                branch_local_3_part_2 = branch_local_3[:, branch_local_2.shape[1]//3:2*branch_local_2.shape[1]//3]
                branch_local_3_part_3 = branch_local_3[:, 2*branch_local_2.shape[1]//3:]

                pool_local_3 = self.pool(branch_local_3, training=training)
                pool_local_3_part_1 = self.pool(branch_local_3_part_1, training=training)
                pool_local_3_part_2 = self.pool(branch_local_3_part_2, training=training)
                pool_local_3_part_3 = self.pool(branch_local_3_part_3, training=training)

                stacked = K.stack([branch_global, pool_local_2, pool_local_2_part_1, pool_local_2_part_2, pool_local_3, pool_local_3_part_1, pool_local_3_part_2, pool_local_3_part_3], axis=1)

                out = self.last([stacked, x[1]], training=training)

                return out

            else:
                out = tf.recompute_grad(self.block_4b)(out, training=training)
                out = tf.recompute_grad(self.block_4c)(out, training=training)
                out = tf.recompute_grad(self.block_4d)(out, training=training)
                out = tf.recompute_grad(self.block_4e)(out, training=training)
                out = tf.recompute_grad(self.block_4f)(out, training=training)
                out = tf.recompute_grad(self.block_5a)(out, training=training)
                out = tf.recompute_grad(self.block_5b)(out, training=training)
                out = tf.recompute_grad(self.block_5c)(out, training=training)
                out = self.pool(out, training=training)
                out = self.last([out, x[1]], training=training)
        else:
            out = self.block_1(x[0], training=training)
            out = self.block_2a(out, training=training)
            out = self.block_2b(out, training=training)
            out = self.block_2c(out, training=training)
            out = self.block_3a(out, training=training)
            out = self.block_3b(out, training=training)
            out = self.block_3c(out, training=training)
            out = self.block_3d(out, training=training)
            out = self.block_4a(out, training=training)

            if mgn:
                branch_global = self.block_4b(out, training=training)
                branch_global = self.block_4b(branch_global, training=training)
                branch_global = self.block_4c(branch_global, training=training)
                branch_global = self.block_4d(branch_global, training=training)
                branch_global = self.block_4e(branch_global, training=training)
                branch_global = self.block_4f(branch_global, training=training)
                branch_global = self.block_5a(branch_global, training=training)
                branch_global = self.block_5b(branch_global, training=training)
                branch_global = self.block_5c(branch_global, training=training)
                branch_global = self.pool(branch_global, training=training)

                branch_local_2 = self.block_4b_local_2(out, training=training)
                branch_local_2 = self.block_4b_local_2(branch_local_2, training=training)
                branch_local_2 = self.block_4c_local_2(branch_local_2, training=training)
                branch_local_2 = self.block_4d_local_2(branch_local_2, training=training)
                branch_local_2 = self.block_4e_local_2(branch_local_2, training=training)
                branch_local_2 = self.block_4f_local_2(branch_local_2, training=training)
                branch_local_2 = self.block_5a_local_2(branch_local_2, training=training)
                branch_local_2 = self.block_5b_local_2(branch_local_2, training=training)
                branch_local_2 = self.block_5c_local_2(branch_local_2, training=training)
                branch_local_2_part_1 = branch_local_2[:, :branch_local_2.shape[1] // 2]
                branch_local_2_part_2 = branch_local_2[:, branch_local_2.shape[1] // 2:]
                pool_local_2 = self.pool(branch_local_2, training=training)
                pool_local_2_part_1 = self.pool(branch_local_2_part_1, training=training)
                pool_local_2_part_2 = self.pool(branch_local_2_part_2, training=training)

                branch_local_3 = self.block_4b_local_3(out, training=training)
                branch_local_3 = self.block_4b_local_3(branch_local_3, training=training)
                branch_local_3 = self.block_4c_local_3(branch_local_3, training=training)
                branch_local_3 = self.block_4d_local_3(branch_local_3, training=training)
                branch_local_3 = self.block_4e_local_3(branch_local_3, training=training)
                branch_local_3 = self.block_4f_local_3(branch_local_3, training=training)
                branch_local_3 = self.block_5a_local_3(branch_local_3, training=training)
                branch_local_3 = self.block_5b_local_3(branch_local_3, training=training)
                branch_local_3 = self.block_5c_local_3(branch_local_3, training=training)
                branch_local_3_part_1 = branch_local_3[:, :branch_local_2.shape[1] // 3]
                branch_local_3_part_2 = branch_local_3[:, branch_local_2.shape[1] // 3:2 * branch_local_2.shape[1] // 3]
                branch_local_3_part_3 = branch_local_3[:, 2 * branch_local_2.shape[1] // 3:]
                pool_local_3 = self.pool(branch_local_3, training=training)
                pool_local_3_part_1 = self.pool(branch_local_3_part_1, training=training)
                pool_local_3_part_2 = self.pool(branch_local_3_part_2, training=training)
                pool_local_3_part_3 = self.pool(branch_local_3_part_3, training=training)

                stacked = K.stack([branch_global, pool_local_2, pool_local_2_part_1, pool_local_2_part_2, pool_local_3,
                                   pool_local_3_part_1, pool_local_3_part_2, pool_local_3_part_3], axis=1)

                out = self.last([stacked, x[1]], training=training)

                return out

            else:
                out = self.block_4b(out, training=training)
                out = self.block_4c(out, training=training)
                out = self.block_4d(out, training=training)
                out = self.block_4e(out, training=training)
                out = self.block_4f(out, training=training)
                out = self.block_5a(out, training=training)
                out = self.block_5b(out, training=training)
                out = self.block_5c(out, training=training)
                out = self.pool(out, training=training)
                out = self.last([out, x[1]], training=training)
        return out
