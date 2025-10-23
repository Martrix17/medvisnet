"""Callbacks for training process such as Early Stopping."""


class EarlyStopping:
    """
    Early stopping to monitor validation loss and stop training when no improvement
    is seen.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.01,
    ) -> None:
        """
        Args:
            patience (int): Number of epochs with no improvement after which training
            will be stopped.
            delta (float): Minimum change in the monitored quantity to qualify as an
            improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Call method to check if early stepping should trigger."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True
        return self.early_stop
