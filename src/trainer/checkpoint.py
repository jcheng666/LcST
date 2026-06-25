"""Best-model checkpoint tracking, early stopping, model save/restore."""

import copy


class CheckpointManager:
    """Tracks the best validation loss and manages model checkpointing."""

    def __init__(self, model, patience):
        self.patience = patience
        self.counter = 0
        self.best_loss = 1e9
        self.best_state = copy.deepcopy(model.grad_state_dict())

    def update(self, val_loss, model):
        """Record a validation loss. Returns True if it's a new best."""
        if val_loss < self.best_loss:
            self.counter = 0
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.grad_state_dict())
            return True
        self.counter += 1
        return False

    def should_stop(self):
        return self.counter >= self.patience

    def restore(self, model):
        """Load the best state back into the model."""
        model.load_state_dict(self.best_state, strict=False)

    def save(self, model, modelpath):
        """Save and reload the model from disk if modelpath is given."""
        if modelpath is not None:
            model.save(modelpath)
            model.load(modelpath)
