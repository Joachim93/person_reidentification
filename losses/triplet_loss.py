import tensorflow as tf
from tensorflow.keras import backend as K
import numbers


class TripletLoss:
    """
    Implementation of Triplet Hard Loss (https://arxiv.org/pdf/1703.07737.pdf)
    Inspired by original implementation from https://github.com/VisualComputingInstitute/triplet-reid

    Input Layers:
        [features]

    Input:
        margin_type : str
            possible values: ['hard', 'soft']
        margin_value : float
            margin value used if margin_type == 'hard'

    Return:
        loss
    """

    def __init__(self, margin_type, margin_value):
        if margin_type == "soft":
            self.margin = "soft"
        else:
            self.margin = margin_value

    @staticmethod
    def all_diffs(a, b):
        """ Returns a tensor of all combinations of a - b.
        Args:
            a (2D tensor): A batch of vectors shaped (B1, F).
            b (2D tensor): A batch of vectors shaped (B2, F).
        Returns:
            The matrix of all pairwise differences between all vectors in `a` and in
            `b`, will be of shape (B1, B2).
        Note:
            For convenience, if either `a` or `b` is a `Distribution` object, its
            mean is used.
        """
        return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

    def cdist(self, a, b, metric='euclidean'):
        """Similar to scipy.spatial's cdist, but symbolic.
        The currently supported metrics can be listed as `cdist.supported_metrics` and are:
            - 'euclidean', although with a fudge-factor epsilon.
            - 'sqeuclidean', the squared euclidean.
            - 'cityblock', the manhattan or L1 distance.
        Args:
            a (2D tensor): The left-hand side, shaped (B1, F).
            b (2D tensor): The right-hand side, shaped (B2, F).
            metric (string): Which distance metric to use, see notes.
        Returns:
            The matrix of all pairwise distances between all vectors in `a` and in
            `b`, will be of shape (B1, B2).
        Note:
            When a square root is taken (such as in the Euclidean case), a small
            epsilon is added because the gradient of the square-root at zero is
            undefined. Thus, it will never return exact zero in these cases.
        """
        with tf.name_scope("cdist"):
            diffs = self.all_diffs(a, b)
            if metric == 'sqeuclidean':
                return tf.reduce_sum(tf.square(diffs), axis=-1)
            elif metric == 'euclidean':
                return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
            elif metric == 'cityblock':
                return tf.reduce_sum(tf.abs(diffs), axis=-1)
            else:
                raise NotImplementedError(
                    'The following metric is not implemented by `cdist` yet: {}'.format(metric))

    def batch_hard(self, dists, pids):
        """Computes the batch-hard loss from arxiv.org/abs/1703.07737.
        Args:
            dists (2D tensor): A square all-to-all distance matrix as given by cdist.
            pids (1D tensor): The identities of the entries in `batch`, shape (B,).
                This can be of any type that can be compared, thus also a string.
            margin: The value of the margin if a number, alternatively the string
                'soft' for using the soft-margin formulation, or `None` for not
                using a margin at all.
        Returns:
            A 1D tensor of shape (B,) containing the loss value for each sample.
        """

        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        positive_mask = tf.math.logical_xor(same_identity_mask,
                                            tf.eye(tf.shape(pids)[0], dtype=tf.bool))

        furthest_positive = tf.reduce_max(dists * tf.cast(positive_mask,
                                                          tf.float32), axis=1)
        closest_negative = tf.reduce_min(dists + 1e5 * tf.cast(same_identity_mask, tf.float32), axis=1)

        diff = furthest_positive - closest_negative
        if isinstance(self.margin, numbers.Real):
            diff = tf.maximum(diff + self.margin, 0.0)
        elif self.margin == 'soft':
            diff = tf.nn.softplus(diff)
        elif self.margin.lower() == 'none':
            pass
        else:
            raise NotImplementedError(
                'The margin {} is not implemented in batch_hard'.format(self.margin))
        return diff

    def compute_loss(self, target, features):
        dists = self.cdist(features, features)
        return K.mean(self.batch_hard(dists, target))
