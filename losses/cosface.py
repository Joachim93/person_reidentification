import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


class Cosface(keras.layers.Layer):
    """
    Implementation of Cosface Loss (https://arxiv.org/pdf/1801.05599.pdf)
    Code is inspired by the reimplementation from https://github.com/KevinMusgrave/pytorch-metric-learning

    Input Layers:
        [features, labels]

    Input:
        margin : float
            margin parameter in Cosface
        scale : int
            scale parameter in Cosface

    Return:
        logits
    """
    def __init__(self, scale=30, margin=0.35, kernel_regularizer=None, **kwargs):
        self.scale = scale
        self.margin = margin
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(Cosface, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_lmsl = self.add_weight(name='cosface_weights',
                                              shape=(input_shape[1][1],  # Num_Classes
                                                     input_shape[0][1]),  # Feature length
                                            initializer="he_uniform",
                                            trainable=True,
                                            regularizer=self.kernel_regularizer)
        super(Cosface, self).build(input_shape)

    def call(self, x):

        normed_weights = tf.nn.l2_normalize(self.weights_lmsl, 1,
                                            name='lmsl_weights_norm')
        normed_features = tf.nn.l2_normalize(x[0], 1,
                                             name='features_norm')

        cosine = K.dot(normed_features, tf.transpose(normed_weights))
        cosine_of_target_classes = cosine[x[1] == 1]

        modified_cosine_target_classes = cosine_of_target_classes - self.margin
        diff = tf.expand_dims(modified_cosine_target_classes-cosine_of_target_classes, axis=1)

        logits = cosine + x[1] * diff

        logits = logits * self.scale

        return logits