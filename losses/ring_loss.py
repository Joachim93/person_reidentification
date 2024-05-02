import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Constant


def squared_ring_loss(x, ring_norm):
    """
    Calculates the squared error of ||x||2 and ring_norm

    Parameters
    ----------
    x : tensor
        Feature vector of the model
    ring_norm : tensor
        Weight of the ring norm

    Returns
    -------
        squared error (SE or L2 Error)
    """
    l2_norm = K.sqrt(K.sum(K.square(x), axis=-1))
    return 0.5 * K.square(l2_norm - ring_norm)


def huber_ring_loss(x, ring_norm, huber_delta=1.0):
    """
    Similar to squared_ring_loss but calculates the hubert error instead of SE
    """
    l2_norm = K.sqrt(K.sum(K.square(x), axis=-1))
    error = l2_norm - ring_norm
    huber_loss = K.switch(error < huber_delta, 0.5 * error ** 2,
                          huber_delta * (error - 0.5 * huber_delta))
    return huber_loss


def cauchy_ring_loss(x, ring_norm, scale_factor=2.3849):
    """
    Similar to squared_ring_loss but calculates the cauchy error instead of SE
    Cauchy constant (scale_factor) taken from:
    http://webdiis.unizar.es/~jcivera/papers/concha_civera_ecmr15.pdf
    """
    alpha = 0.5 * (scale_factor ** 2)
    l2_norm = K.sqrt(K.sum(K.square(x), axis=-1))
    return alpha * K.log(1.0 + K.square((l2_norm - ring_norm) / scale_factor))


def geman_ring_loss(x, ring_norm, alpha=0.5):
    """
    Similar to squared_ring_loss but calculates the geman error instead of SE
    Geman constant (alpha) taken from:
    http://webdiis.unizar.es/~jcivera/papers/concha_civera_ecmr15.pdf
    """
    l2_norm = K.sqrt(K.sum(K.square(x), axis=-1))
    squared_error = K.square(l2_norm - ring_norm)
    return tf.divide(alpha * squared_error, squared_error + (2.0 * alpha))


class RingLoss(keras.layers.Layer):
    """
    Implementation of Ring Loss (https://arxiv.org/pdf/1803.00130.pdf)

    Input Layer:
        [feature]

    Input:
        radius : float
            Radius in Ring Loss (constant with which the ring_norm is multiplied)
        loss_type : loss_type_name
            possible: ['squared', 'cauchy', 'geman', 'huber']
            See http://webdiis.unizar.es/~jcivera/papers/concha_civera_ecmr15.pdf
            for an explanation to the different loss types.

    Output:
        Ring Loss
    """
    def __init__(self, radius=1.0, loss_type='squared', **kwargs):
        self.radius = radius
        self.huber_delta = 1.0
        self.cauchy_scale_factor = 2.3849  # cauchy constant
        self.geman_alpha = 0.5  # geman constant
        self.loss_type = loss_type
        super(RingLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        self.var_shape = (1,)
        # init norm value
        self.ring_norm = self.add_weight(name='ring_norm',
                                         shape=self.var_shape,
                                         initializer=Constant(self.radius),
                                         dtype=K.floatx(),
                                         trainable=self.trainable)
        super(RingLoss, self).build(input_shape)

    def call(self, x):
        if self.loss_type == 'squared':
            # calculate L2 ring loss
            self.ring_loss = squared_ring_loss(x, self.ring_norm)

        elif self.loss_type == 'cauchy':
            # calculate cauchy ring loss
            self.ring_loss = cauchy_ring_loss(x, self.ring_norm,
                                              scale_factor=self.cauchy_scale_factor)

        elif self.loss_type == 'geman':
            # calculate geman-mcclure ring loss
            self.ring_loss = geman_ring_loss(x, self.ring_norm,
                                             alpha=self.geman_alpha)

        elif self.loss_type == 'huber':
            # calculate Smooth-L1/Huber ring loss
            self.ring_loss = huber_ring_loss(x, self.ring_norm,
                                             huber_delta=self.huber_delta)
        else:
            print(
                "Invalid Loss Name specified. Available options are : 'squared', 'huber', 'cauchy' and 'geman'. Continuing with default loss - 'huber'")
            self.ring_loss = huber_ring_loss(x, self.ring_norm,
                                             huber_delta=self.huber_delta)

        return K.mean(self.ring_loss)