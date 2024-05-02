"""
Script for Resnet50 architecture with pretrained weights from keras
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.utils.data_utils import get_file

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_format = 'channels_last'
keras.backend.set_image_data_format(data_format)

if K.image_data_format() == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1

weights_path_no_top = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path = get_file(
          'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
          weights_path_no_top,
          cache_subdir='models',
          md5_hash='a268eb855778b3df3c7506639542a6af')


def identity_block(input_shape, kernel_size, filters, stage, block, decay, path=weights_path):
    """The identity block is the block that has no conv layer at shortcut.

    Arguments:
      input_shape: input shape of images
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      decay: weight decay factor to use for all layers
      path: path to pretrained weights

    Returns:
      keras model
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(decay), bias_regularizer=l2(decay),
                      name=conv_name_base + '2a')(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', beta_regularizer=l2(decay), gamma_regularizer=l2(decay))(x)

    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2, kernel_size, kernel_initializer='he_normal', kernel_regularizer=l2(decay), bias_regularizer=l2(decay),
        padding='same', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', beta_regularizer=l2(decay), gamma_regularizer=l2(decay))(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(decay), bias_regularizer=l2(decay),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', beta_regularizer=l2(decay), gamma_regularizer=l2(decay))(x)

    x = layers.add([x, inputs])
    x = layers.Activation('relu')(x)
    model = keras.Model(inputs, x)
    if path:
        model.load_weights(path, by_name=True)
    return model


def conv_block(input_shape, kernel_size, filters, stage, block, decay, strides=(2, 2), path=weights_path):
    """A block that has a conv layer at shortcut.

    Arguments:
      input_shape: input shape of images
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      decay: weight decay factor to use for all layers
      strides: Strides for the first conv layer in the block
      path: path to pretrained weights

    Returns:
      keras model

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(
        filters1, (1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer=l2(decay), bias_regularizer=l2(decay),
        name=conv_name_base + '2a')(
        inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', beta_regularizer=l2(decay), gamma_regularizer=l2(decay))(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(decay), bias_regularizer=l2(decay),
        name=conv_name_base + '2b')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', beta_regularizer=l2(decay), gamma_regularizer=l2(decay))(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(decay), bias_regularizer=l2(decay),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', beta_regularizer=l2(decay), gamma_regularizer=l2(decay))(x)

    shortcut = layers.Conv2D(
        filters3, (1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer=l2(decay), bias_regularizer=l2(decay),
        name=conv_name_base + '1')(inputs)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1', beta_regularizer=l2(decay), gamma_regularizer=l2(decay))(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    model = keras.Model(inputs, x)
    if path:
        model.load_weights(path, by_name=True)
    return model


def first_block(input_shape, decay, path=weights_path):
    """
    First block of ResNet50.

    Arguments:
      input_shape: input shape of images
      decay: weight decay factor to use for all layers
      path: path to pretrained weights

    Returns:
      keras model
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(
        64, (7, 7), strides=(2, 2), kernel_initializer='he_normal', kernel_regularizer=l2(decay), bias_regularizer=l2(decay),
        padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1', beta_regularizer=l2(decay), gamma_regularizer=l2(decay))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    model = keras.Model(inputs, x, name="first_block")
    if path:
        model.load_weights(path, by_name=True)
    return model


class ResNetKeras(Model):

    def __init__(self, args, input_shape, output_layers, decay, stride=2, mgn=False):
        super(ResNetKeras, self).__init__()
        self.block_1 = first_block(input_shape, decay=decay)
        self.block_2a = conv_block(self.block_1.output_shape[1:], 3, [64, 64, 256], stage=2, block='a', decay=decay,
                                   strides=(1, 1))
        self.block_2b = identity_block(self.block_2a.output_shape[1:], 3, [64, 64, 256], stage=2, decay=decay,
                                       block='b')
        self.block_2c = identity_block(self.block_2b.output_shape[1:], 3, [64, 64, 256], stage=2, decay=decay,
                                       block='c')

        self.block_3a = conv_block(self.block_2c.output_shape[1:], 3, [128, 128, 512], stage=3, block='a', decay=decay)
        self.block_3b = identity_block(self.block_3a.output_shape[1:], 3, [128, 128, 512], stage=3, block='b',
                                       decay=decay)
        self.block_3c = identity_block(self.block_3b.output_shape[1:], 3, [128, 128, 512], stage=3, block='c',
                                       decay=decay)
        self.block_3d = identity_block(self.block_3c.output_shape[1:], 3, [128, 128, 512], stage=3, block='d',
                                       decay=decay)

        self.block_4a = conv_block(self.block_3d.output_shape[1:], 3, [256, 256, 1024], stage=4, block='a', decay=decay)
        self.block_4b = identity_block(self.block_4a.output_shape[1:], 3, [256, 256, 1024], stage=4, block='b',
                                       decay=decay)
        self.block_4c = identity_block(self.block_4b.output_shape[1:], 3, [256, 256, 1024], stage=4, block='c',
                                       decay=decay)
        self.block_4d = identity_block(self.block_4c.output_shape[1:], 3, [256, 256, 1024], stage=4, block='d',
                                       decay=decay)
        self.block_4e = identity_block(self.block_4d.output_shape[1:], 3, [256, 256, 1024], stage=4, block='e',
                                       decay=decay)
        self.block_4f = identity_block(self.block_4e.output_shape[1:], 3, [256, 256, 1024], stage=4, block='f',
                                       decay=decay)

        self.block_5a = conv_block(self.block_4f.output_shape[1:], 3, [512, 512, 2048], stage=5, block='a', decay=decay,
                                   strides=stride)
        self.block_5b = identity_block(self.block_5a.output_shape[1:], 3, [512, 512, 2048], stage=5, block='b',
                                       decay=decay)
        self.block_5c = identity_block(self.block_5b.output_shape[1:], 3, [512, 512, 2048], stage=5, block='c',
                                       decay=decay)

        if mgn:
            self.block_4b_local_2 = identity_block(self.block_4a.output_shape[1:], 3, [256, 256, 1024], stage=4,
                                                   block='b', decay=decay)
            self.block_4c_local_2 = identity_block(self.block_4b_local_2.output_shape[1:], 3, [256, 256, 1024], stage=4,
                                                   block='c', decay=decay)
            self.block_4d_local_2 = identity_block(self.block_4c_local_2.output_shape[1:], 3, [256, 256, 1024], stage=4,
                                                   block='d', decay=decay)
            self.block_4e_local_2 = identity_block(self.block_4d_local_2.output_shape[1:], 3, [256, 256, 1024], stage=4,
                                                   block='e', decay=decay)
            self.block_4f_local_2 = identity_block(self.block_4e_local_2.output_shape[1:], 3, [256, 256, 1024], stage=4,
                                                   block='f', decay=decay)

            self.block_5a_local_2 = conv_block(self.block_4f_local_2.output_shape[1:], 3, [512, 512, 2048], stage=5,
                                               block='a', decay=decay,
                                               strides=1)
            self.block_5b_local_2 = identity_block(self.block_5a_local_2.output_shape[1:], 3, [512, 512, 2048], stage=5,
                                                   block='b', decay=decay)
            self.block_5c_local_2 = identity_block(self.block_5b_local_2.output_shape[1:], 3, [512, 512, 2048], stage=5,
                                                   block='c', decay=decay)

            self.block_4b_local_3 = identity_block(self.block_4a.output_shape[1:], 3, [256, 256, 1024], stage=4,
                                                   block='b', decay=decay)
            self.block_4c_local_3 = identity_block(self.block_4b_local_3.output_shape[1:], 3, [256, 256, 1024], stage=4,
                                                   block='c', decay=decay)
            self.block_4d_local_3 = identity_block(self.block_4c_local_3.output_shape[1:], 3, [256, 256, 1024], stage=4,
                                                   block='d', decay=decay)
            self.block_4e_local_3 = identity_block(self.block_4d_local_3.output_shape[1:], 3, [256, 256, 1024], stage=4,
                                                   block='e', decay=decay)
            self.block_4f_local_3 = identity_block(self.block_4e_local_3.output_shape[1:], 3, [256, 256, 1024], stage=4,
                                                   block='f', decay=decay)

            self.block_5a_local_3 = conv_block(self.block_4f_local_3.output_shape[1:], 3, [512, 512, 2048], stage=5,
                                               block='a', decay=decay,
                                               strides=1)
            self.block_5b_local_3 = identity_block(self.block_5a_local_3.output_shape[1:], 3, [512, 512, 2048], stage=5,
                                                   block='b', decay=decay)
            self.block_5c_local_3 = identity_block(self.block_5b_local_3.output_shape[1:], 3, [512, 512, 2048], stage=5,
                                                   block='c', decay=decay)

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
                branch_local_2_part_1 = branch_local_2[:, :branch_local_2.shape[1] // 2]
                branch_local_2_part_2 = branch_local_2[:, branch_local_2.shape[1] // 2:]

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
