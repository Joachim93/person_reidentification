from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import regularizers


class CircleLoss(keras.layers.Layer):
    """
    Implementation of Circle Loss (https://arxiv.org/abs/2002.10857)
    The code is inspired by the reimplementation from https://github.com/JDAI-CV/fast-reid

    Input Layers:
        [features, labels]

    Input:
        margin : float
            margin parameter in Circle Loss
        scale : int
            scale parameter in Circle Loss

    Return:
        logits

    Notes:
        This code implements Circle Loss as classification loss, where the similarities are calculated
        with respect to the class labels. It is also possible to use Circle Loss as a metric loss and
        calculate similarities between all feature vectors of a batch (implemented in pairwise_circle.py).
    """
    def __init__(self, scale=128, margin=0.25, kernel_regularizer=None, **kwargs):
        self.scale = scale
        self.margin = margin
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(CircleLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_circle = self.add_weight(name='circle_loss_weights',
                                              shape=(input_shape[1][1],  # Num_Classes
                                                     input_shape[0][1]),  # Feature length
                                            initializer="he_uniform",
                                            trainable=True,
                                            regularizer=self.kernel_regularizer)
        super(CircleLoss, self).build(input_shape)

    def call(self, x):

        normed_weights = tf.nn.l2_normalize(self.weights_circle, 1,
                                            name='circle_loss_weights_norm')
        normed_features = tf.nn.l2_normalize(x[0], 1,
                                             name='features_norm')

        similarities = K.dot(normed_features, tf.transpose(normed_weights))

        alpha_p = K.clip(-tf.stop_gradient(similarities) + 1 + self.margin, min_value=0, max_value=None)
        alpha_n = K.clip(tf.stop_gradient(similarities) + self.margin, min_value=0, max_value=None)

        delta_p = 1 - self.margin
        delta_n = self.margin

        s_p = self.scale * alpha_p * (similarities - delta_p)
        s_n = self.scale * alpha_n * (similarities - delta_n)

        targets = x[1]

        logits = targets * s_p + (1.0 - targets) * s_n

        return logits
