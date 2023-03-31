import numpy as np

class EarlyStopping():
    """Stop training when a monitored quantity (training or testing loss) has stopped improving.
    Only checked at validation step according to ``display_every`` in ``Model.train``.

    Args:
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        monitor: The loss function that is monitored. Either 'loss_train' or 'loss_test'
    """

    def __init__(self, model, min_delta=0, patience=0):
        self.model = None
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.model = model

        self.monitor_op = np.less
        self.min_delta *= -1

        self.stop_training = False

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, loss):
        current = loss
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print("Epoch {}: early stopping".format(self.stopped_epoch))