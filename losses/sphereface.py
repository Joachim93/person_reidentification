import tensorflow as tf
from tensorflow import keras
import scipy.special
from tensorflow.keras import regularizers
import math
from tensorflow.keras import backend as K


class Sphereface(keras.layers.Layer):
    """
    Implementation of Sphereface Loss (https://arxiv.org/pdf/1704.08063.pdf)
    Code is inspired by the reimplementation from https://github.com/KevinMusgrave/pytorch-metric-learning

    Input Layers:
        [features, labels]

    Input:
        margin : float
            margin parameter in Sphereface
        scale : int
            scale parameter in Sphereface

    Return:
        logits
    """

    def __init__(self, scale=14, margin=1, kernel_regularizer=None, **kwargs):
        self.scale = scale
        self.margin = int(margin)
        self.max_n = self.margin // 2
        self.n_range = tf.constant([n for n in range(0, self.max_n+1)], dtype=tf.float32)
        self.margin_choose_n = tf.constant(
            [scipy.special.binom(self.margin, 2 * n) for n in self.n_range], dtype=tf.float32
        )
        self.cos_powers = tf.constant([self.margin - (2 * n.numpy()) for n in self.n_range], dtype=tf.float32)
        self.alternating = tf.constant([(-1) ** n.numpy() for n in self.n_range], dtype=tf.float32)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(Sphereface, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_lmsl = self.add_weight(name='large_margin_softmax_loss_weights',
                                              shape=(input_shape[1][1],  # Num_Classes
                                                     input_shape[0][1]),  # Feature length
                                            initializer="he_uniform",
                                            trainable=True,
                                            regularizer=self.kernel_regularizer)
        super(Sphereface, self).build(input_shape)

    def call(self, x):

        normed_weights = tf.nn.l2_normalize(self.weights_lmsl, 1,
                                            name='lmsl_weights_norm')
        normed_features = tf.nn.l2_normalize(x[0], 1,
                                             name='features_norm')

        cosine = K.dot(normed_features, tf.transpose(normed_weights))
        cosine_of_target_classes = cosine[x[1] == 1]
        cosine_of_target_classes_expanded = tf.expand_dims(cosine_of_target_classes, axis=1)

        cos_powered = cosine_of_target_classes_expanded ** self.cos_powers
        # sin_powered = (1-cosine_of_target_classes_expanded**2) ** self.n_range
        sin_powered = tf.pow(1 - cosine_of_target_classes_expanded ** 2, 0.5) ** self.n_range

        terms = self.alternating * self.margin_choose_n * cos_powered * sin_powered
        cos_with_margin = tf.math.reduce_sum(terms, axis=1)

        angles = tf.math.acos(tf.clip_by_value(cosine_of_target_classes, -1, 1))
        k = tf.stop_gradient(tf.math.floor(angles / (math.pi / self.margin)))
        modified_cosine_target_classes = ((-1) ** k) * cos_with_margin - (2 * k)
        diff = tf.expand_dims(modified_cosine_target_classes-cosine_of_target_classes, axis=1)

        logits = cosine + x[1] * diff

        embedding_norm = tf.norm(normed_features, axis=1)
        weights_norm = tf.norm(normed_weights, axis=1)
        product_of_magnitudes = tf.expand_dims(weights_norm, axis=0) * tf.expand_dims(embedding_norm, axis=1)
        # logits = logits * self.scale
        logits = logits * product_of_magnitudes * self.scale


        return logits