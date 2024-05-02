from tensorflow import keras
import tensorflow.keras.backend as K


class CenterLoss(keras.layers.Layer):
    """
    Implementation of Center Loss (https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31)

    Input Layers:
        [features, labels]

    Input:
        num_classes : int
            number of different classes contained in the dataset
        feat_dim : int
            length of the feature vector

    Return:
        loss

    Notes:
        The class center aren't calculated explicit from the data. Instead they are randomly initialised
        and optimized with SGD after each update step.
    """
    def __init__(self, num_classes=751, feat_dim=2048):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                         shape=(self.num_classes, self.feat_dim),
                                         initializer=keras.initializers.RandomNormal(stddev=1),
                                         trainable=True)
        super(CenterLoss, self).build(input_shape)

    def call(self, data):

        x = data[0]
        labels = data[1]
        if x.shape[0] == labels.shape[0]:
            result = x - K.dot(labels, self.centers)
            result = K.sum(result**2, axis=1, keepdims=True)
            result = K.clip(result, 1e-12, 1e+12)
            return K.mean(result)
        else:
            return 0



