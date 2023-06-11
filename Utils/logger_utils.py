import torch
import os
from datetime import timedelta
import yaml

class TrainingLogger:
    """
    Training logger that logs,saves and prints model / training data
    """
    def __init__(self, settings, world_size=1):
        self.settings = settings
        self.worldsize = world_size
        
        self.checkpoint_path = settings.get("SAVE_PATH_CHECKPOINT")
        self.best_path = settings.get("SAVE_PATH_BEST")
        self.log_path = os.path.join(self.checkpoint_path, "log.txt")
        self.settings_save_path = os.path.join(self.checkpoint_path, "settings.yaml")
        
        try:
            os.makedirs(self.checkpoint_path)
            os.makedirs(self.best_path)
        except:
            print("FILES ALREADY CREATED", flush=True)
        
        
        self.total_epochs = settings.get("EPOCHS")
        self.lr = settings.get("L_RATE")
        
        self.log_entry = settings.get("LOG_ENTRY")
        self.save_entry = settings.get("SAVE_ENTRY")
        
        self.current_epoch = 0
        self.current_loss = 1e6
        self.loss_dict = {}
        
        self.eval_metrics = {"map": -1}
        
        self.best_map = -1
        
        self.duration = 0
        #self.init_print_settings()

    
    def init_print_settings(self):
        """
        Print the settings for the training
        """
        pp_ = "="*30
        print(f"{pp_}Settings{pp_}",flush=True)
        print("|>>>{0}--{1}\n".format(self.settings.get("BACKBONE"), self.settings.get("DATASET")),flush=True)
        print(f"|>>>NUM GPUS: {self.worldsize}\n", flush=True)
        print(f"|>>>EPOCHS: {self.total_epochs}\n",flush=True)
        print(f"|>>>INITIAL LEARNING RATE: {self.lr}\n",flush=True)
        print("|>>>BATCH: {0} | VALIDATION BATCH: {1}\n".format(self.settings.get("BATCH"), self.settings.get("VAL_BATCH")),flush=True)
        print("|>>>MIN IMAGE SIZE: {0} | MAX IMAGE SIZE: {1}\n".format(self.settings.get("MIN_SIZE"), self.settings.get("MAX_SIZE")),flush=True)
        print("|>>>SEPERATE TRAINING: {0}\n".format(self.settings.get("SEPERATE_TRAIN")), flush=True)
        print(f"{pp_}========{pp_}",flush=True)
    
    def getSaveDict(self, model, optimizer):
        """
        Get the save dict
        """
        try:
            model_dict = model.module.state_dict()
        except:
            model_dict = model.state_dict()
        
        return {
            "epoch": self.current_epoch,
            "loss_dict": self.loss_dict,
            "loss": self.current_loss,
            "mAP": self.eval_metrics['map'],
            "model_dict": model_dict,
            "optimizer_dict": optimizer.state_dict()
        }
    
    def bestSave(self, model, optimizer):
        """
        Save best model
        """
        save_dict = self.getSaveDict(model, optimizer)
        
        torch.save(save_dict, os.path.join(self.best_path,f"best_model.pt"))
        print("Best Model Saved", flush=True)
    
    def intervalSave(self, model, optimizer):
        """
        Save model at interval
        """
        save_dict = self.getSaveDict(model, optimizer)
        
        torch.save(save_dict, os.path.join(self.checkpoint_path,f"checkpoint_{self.current_epoch}.pt"))
        print(f"Saved: checkpoint_{self.current_epoch}.pt", flush=True)
    
    def log(self):
        """
        Log the data
        """
        with open(self.log_path, "a") as f:
            map_value = self.eval_metrics["map"]
            f.write(f"{self.current_epoch}={self.current_loss}={map_value}\n")
        
    def save(self, model, optimizer):
        """
        Save / log model data
        """
        #Save best model
        if self.eval_metrics['map'] > self.best_map:
            self.best_map = self.eval_metrics['map']
            self.bestSave(model, optimizer)
            self.settings.writeFinalConfig(self.settings_save_path)
        
        #Save at interval
        if self.current_epoch % self.save_entry == 0 or self.current_epoch == self.total_epochs:
            self.intervalSave(model, optimizer)
            self.settings.writeFinalConfig(self.settings_save_path)
        
        #Save log
        if self.current_epoch % self.log_entry == 0:
            self.log()
    
    def summarizeMetrics(self):
        """
        Summarize the evaluation metrics
        """
        if len(self.eval_metrics) > 1:
            metric_string = ["{}:{:.3f}".format(k,v*100) for k,v in self.eval_metrics.items()]
            print("\n".join(metric_string), flush=True)

    def summarize(self):
        """
        Summarize and print the training data per epoch
        """
        print(f"<<<{self.current_epoch}>>> Loss: {self.current_loss} | LR: {self.lr} | Duration: {str(timedelta(seconds = self.duration))}",flush=True)
        
        if len(self.loss_dict) > 0:
            printLosses = [f"{k}:{v}" for k,v in self.loss_dict.items()]
            print("<<<{0}>>> {1}".format(self.current_epoch, " | ".join(printLosses)), flush=True)
        
        self.summarizeMetrics()
        
    def update(self, losses, epoch, duration, model, optimizer, eval_metric={"map": -1}, losses_dict={}):
        """
        Update values per epoch
        """
        self.current_loss = losses
        self.loss_dict = losses_dict
        self.current_epoch = epoch
        self.eval_metrics = eval_metric
        self.lr = optimizer.param_groups[0]['lr']
        self.duration = duration
        
        self.save(model, optimizer)
        
        
