import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import math
from tensorflow.keras import backend as K
import numpy as np


class AAML(keras.layers.Layer):
    """
    Implementation of Additive Angular Margin Loss (https://arxiv.org/pdf/1801.07698.pdf)
    Original paper code (MxNet): https://github.com/deepinsight/insightface/blob/master/src/train_softmax.py#L206

    Input Layers:
        [features, labels]

    Input:
        margin : float
            margin parameter in AAML
        easy_margin : bool
            If False: AAML is calculated like in the paper
            (otherwise see notes)
        scale : int
            scale parameter in AAML

    Return:
        logits

    Notes:
        For easy margin:
        Every logit (cos(theta)) with a theta higher than 90 will not be changed
        IT is 90 and not x with 0=cos(x+margin), cause tf.to_float(...) compares
        cos(theta) with 0 and not cos(theta+margin) with 0 !
        The original code is therefore buggy. the arcface logit will jump to the
        softmax logit if theta is higher than 90. this results in a higher logit
        starting at 90.

        For hard margin:
        I believe mm and threshold are calculated in such a way, that the
        logits right below the threshold will not be better than the logits
        right above the threshold. So that the error always falls.
    """

    def __init__(self, margin=0.5, easy_margin=False, scale=64, kernel_regularizer=None, **kwargs):
        assert 0.0 <= margin < np.pi / 2
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.scale = scale
        self.easy_margin = easy_margin
        # For hard margin
        self.mm = math.sin(math.pi - margin) * margin
        self.threshold = math.cos(math.pi - margin)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        super(AAML, self).__init__(**kwargs)

    def build(self, input_shape):
        # init norm value
        self.kernel = self.add_weight(name='aaml_weights',
                                      shape=(input_shape[1][1],  # Num_Classes
                                             input_shape[0][1]),  # Feature length
                                      initializer="he_uniform",
                                      dtype=K.floatx(),
                                      trainable=self.trainable,
                                      regularizer=self.kernel_regularizer)
        super(AAML, self).build(input_shape)

    def call(self, x):
        normed_weights = tf.nn.l2_normalize(self.kernel, 1, 1e-10,
                                            name='aaml_weights_norm')
        normed_features = tf.nn.l2_normalize(x[0], 1, 1e-10,
                                             name='features_norm')

        cos = tf.matmul(normed_features, normed_weights,
                        transpose_a=False, transpose_b=True, name='cos')
        cos_2 = tf.pow(cos, 2., name='cos_2')
        sin = tf.pow(1. - cos_2, .5, name='sin')
        cos_theta_m = self.scale * (self.cos_m * cos
                                    - self.sin_m * sin) * x[1]

        if self.easy_margin:
            # This is like in the original implementation. But this is not a
            # good solution for a "easy_margin". See notes at the top.
            clip_mask = tf.cast(cos >= 0., tf.float32) * x[1]
            clip_out = self.scale * cos * x[1]
            logits = self.scale * cos * (1. - x[1]) + tf.where(clip_mask > 0.,
                                                               cos_theta_m,
                                                               clip_out)
        else:
            clip_mask = tf.cast(cos >= self.threshold, tf.float32) * x[1]
            clip_out = self.scale * (cos - self.mm) * x[1]
            logits = self.scale * cos * (1. - x[1]) + tf.where(clip_mask > 0.,
                                                               cos_theta_m,
                                                               clip_out)
        return logits
