import tensorflow as tf
from tensorflow.keras import backend as K


class PairwiseCircleLoss:
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
        loss

    Notes:
        This code implements Circle Loss as metric loss, where the similarities are calculated between all
        features of a batch. It is also possible to use Circle Loss as a classification loss and
        calculate similarities with respect to the class labels (implemented in circle_loss.py).
    """

    def __init__(self, scale, margin):
        self.scale = scale
        self.margin = margin

    def compute_loss(self, targets, features):
        normed_features = tf.nn.l2_normalize(features, 1)
        dist_mat = K.dot(normed_features, tf.transpose(normed_features))

        same_identity_mask = tf.equal(tf.expand_dims(targets, axis=1),
                                      tf.expand_dims(targets, axis=0))
        negative_mask = tf.cast(tf.math.logical_not(same_identity_mask), tf.float32)
        positive_mask = tf.cast(tf.math.logical_xor(same_identity_mask,
                                                    tf.eye(tf.shape(targets)[0], dtype=tf.bool)), tf.float32)

        s_p = dist_mat * positive_mask
        s_n = dist_mat * negative_mask

        alpha_p = K.clip(-tf.stop_gradient(s_p) + 1 + self.margin, min_value=0, max_value=None)
        alpha_n = K.clip(tf.stop_gradient(s_n) + self.margin, min_value=0, max_value=None)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - self.scale * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - positive_mask)
        logit_n = self.scale * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - negative_mask)

        loss = K.mean(tf.math.softplus(tf.math.reduce_logsumexp(logit_p, axis=1) +
                                       tf.math.reduce_logsumexp(logit_n, axis=1)))

        return loss
