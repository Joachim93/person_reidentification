"""
Functions for creating different learning rate schedulers
"""


def get_lr_scheduler(arguments):
    """
    Decides which learning rate scheduler should be returned.
    """
    if arguments.learning_rate_updates == "epochwise":
        return LearningRateSchedulerEpochs(arguments.start_learning_rate, arguments.warmup_epochs,
                                           arguments.learning_rate_steps, arguments.freeze_epochs)
    else:
        return LearningRateSchedulerBatches(arguments.start_learning_rate, arguments.warmup_epochs)


class LearningRateSchedulerEpochs:
    """
    Learning rate scheduler with epochwise updates which realizes strategy with linear rise for warmup and a
    stepwise decline for fine tuning.

    Parameters
    ----------
    start_lr: float
        learning rate to start with after warmup
        (learning rate at begin of warmup will be 1% of start_lr)
    warmup_epochs: int
        number of epochs for warmup
        (warmup epochs doesn't start counting until freeze epochs are finished)
    steps: list
        epoch where the learning rate is reduced to a tenth
        - currently only one or two elements in this list are supported
        - if learning rate should be constant until the end of training, choose a value bigger than total epochs
    freeze_epochs: int
        number of epochs where the backbone is frozen and only the head of the model is trained

    Returns
    -------
    learning rate: float
        current learning rate

    Notes:
        When using the SGD optimizer an additional exponential decay is applied to the learning rate.
    """

    def __init__(self, start_lr, warmup_epochs, steps, freeze_epochs):
        self.start_lr = start_lr
        self.warmup_epochs = warmup_epochs
        self.steps = steps
        self.freeze_epochs = freeze_epochs

    def get_lr(self, epoch):
        if epoch < (self.freeze_epochs / 2):
            alpha = epoch / (self.freeze_epochs / 2)
            warmup_factor = 0.01 * (1 - alpha) + alpha
            lr = self.start_lr * warmup_factor
        elif epoch < self.freeze_epochs:
            alpha = 1 - (epoch - (self.freeze_epochs/2 - 1)) / (self.freeze_epochs/2)
            warmup_factor = 0.01 * (1 - alpha) + alpha
            lr = self.start_lr * warmup_factor
        elif (epoch >= self.freeze_epochs) & (epoch < self.warmup_epochs+self.freeze_epochs):
            alpha = (epoch - self.freeze_epochs) / self.warmup_epochs
            warmup_factor = 0.01 * (1 - alpha) + alpha
            lr = self.start_lr * warmup_factor
        elif epoch < self.steps[0]:
            lr = self.start_lr
        elif (len(self.steps) == 1) or (epoch < self.steps[1]):
            lr = self.start_lr * 0.1
        else:
            lr = self.start_lr * 0.01
        return lr


class LearningRateSchedulerBatches:
    """
    Learning rate scheduler with epochwise updates which realizes a linear rise for warmup and a linear decay
    for fine tuning.

    Parameters
    ----------
    start_lr: float
        learning rate to start with after warmup
        (learning rate at begin of warmup will be 1% of start_lr)
    warmup_epochs: int
        number of epochs for warmup
        (warmup epochs doesn't start counting until freeze epochs are finished)

    Returns
    -------
    learning rate: float
        current learning rate

    Notes:
        When using the SGD optimizer an additional exponential decay is applied to the learning rate.
    """

    def __init__(self, start_lr, warmup_epochs):
        self.start_lr = start_lr
        self.warmup_epochs = warmup_epochs

    def get_lr(self, epoch, batch, max_batch, max_epochs):
        if epoch < self.warmup_epochs:
            # linear rise from 1/rise start_lr to full start_lr over rise epochs
            lr_start_epoch = (self.start_lr / self.warmup_epochs) * (epoch - 1) + self.start_lr / self.warmup_epochs
            lr_end_epoch = (self.start_lr / self.warmup_epochs) * epoch + self.start_lr / self.warmup_epochs
            lr = lr_start_epoch + (lr_end_epoch - lr_start_epoch) * (batch / max_batch)
            return lr
        else:
            # linear fall
            t = ((-self.start_lr / (max_epochs - self.warmup_epochs)) * -self.warmup_epochs)
            lr_start_epoch = -(self.start_lr / (max_epochs - self.warmup_epochs)) * epoch + self.start_lr + t
            lr_end_epoch = -(self.start_lr / (max_epochs - self.warmup_epochs)) * (epoch + 1) + self.start_lr + t
            lr = lr_start_epoch + (lr_end_epoch - lr_start_epoch) * (batch / max_batch)
            return lr
