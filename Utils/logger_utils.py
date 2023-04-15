

class TrainingLogger:
    
    def __init__(self, checkpoint_path, best_path, total_epochs):
        self.checkpoint_path = checkpoint_path
        self.best_path = best_path
        self.total_epochs = total_epochs