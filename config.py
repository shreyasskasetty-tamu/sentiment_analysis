# configuration
class Config:
    def __init__(self, learning_rate, train_batch_size, valid_batch_size, epochs, model_type, unfreeze_layers=None):
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.epochs = epochs
        self.model_type = model_type
        self.unfreeze_layers = unfreeze_layers
        # Add other relevant config parameters as needed
    
    def pretty_print(self):
        print(f"Configuration:\n"
                f"  Model Type: {self.model_type}\n"
                f"  Learning Rate: {self.learning_rate}\n"
                f"  Training Batch Size: {self.train_batch_size}\n"
                f"  Validation Batch Size: {self.valid_batch_size}\n"
                f"  Epochs: {self.epochs}\n"
                f"  Unfreeze Layers: {'None' if self.unfreeze_layers is None else self.unfreeze_layers}\n")