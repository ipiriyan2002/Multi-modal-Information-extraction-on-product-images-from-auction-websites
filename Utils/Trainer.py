#Import torch packages
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
#Import custom packages
from lib.Model.faster_rcnn_detector import FRCNNDetector
from lib.Loaders.base_dataset import BaseDataset
import Utils.utils as utils
from Utils.eval import evaluate
from Utils.config_loader import ConfigLoader
from Utils.logger_utils import TrainingLogger
#Import other packages
import os, io, time, datetime
import numpy as np


class Trainer():
    def __init__(self, device, settings, with_val=True, world_size=1, resume=False, resume_path=""):
        """
        Arguments:
            device: device
            settigns: settigns
            resume (boolean) : resume from checkpoint
            resume_path : checkpoint path
        """
        #normal settings
        self.loadedSettings = ConfigLoader(settings)
        
        resume_dict = {
            "config":self.loadedSettings.getDict(),
            "epoch": 0,
            "loss": 1e14
                      }
        
        if resume:
            assert len(resume_path) > 1, "To resume, give resume pytorch dict"
            
            resume_dict = torch.load(resume_path)
        
        
        #Make sure we have config name
        self.settings = self.loadedSettings.setDict(resume_dict["config"])
        self.with_val = with_val
        self.device = device
        #Resume point is initially 0
        self.resume_point = resume_dict["epoch"]
        #Previous minimum loss set to 1e14
        self.prev_min_loss = resume_dict["loss"]
        
        #Logging
        self.logger = TrainingLogger(self.settings, world_size)
        if self.device == 0:
            self.logger.init_print_settings()
        
        #Defining training and validation loader
        train_loader = self.getDataLoader(split="train")
        if self.with_val:
            val_loader = self.getDataLoader(split="validation")
        
        #Defining model
        self.model = FRCNNDetector(self.settings)
        self.model = self.model.to(self.device)
        
        #Defining Optimizer
        if self.settings.get("OPT_TYPE") == "ADAM":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.settings.get["L_RATE"], 
                weight_decay=self.settings.get['WEIGHT_DECAY']
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.settings.get('L_RATE'), 
                momentum=self.settings.get('MOMENTUM'), 
                weight_decay=self.settings.get('W_DECAY')
            )
        
        #Define Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.settings.get('STEP_SIZE'), 
            gamma=self.settings.get('GAMMA')
        )
        
        #Defining the Model in terms of DistributedDataParallel
        if self.settings.get("DDP"):
            self.model = DDP(
                self.model, 
                device_ids=[self.device], 
                find_unused_parameters=True
            )
        
        
        #Load the checkpoint details and model,optimizer state dicts
        if resume:
            self.model.load_state_dict(resume_dict['model_dict'])
            self.optimizer.load_state_dict(resume_dict['optimizer_dict'])
        
        
    
    def getDataLoader(self, split="train"):
        
        start_ = time.time()
        
        #Get dataset
        dataset = BaseDataset(self.settings, pad=True, split=split)
        #Get loader
        loader = DataLoader(
            dataset.getDataset(),
            batch_size=self.settings.get("BATCH"),
            shuffle=(split=="train"),
            pin_memory=True,
            num_worker=self.settings.get("NUM_WORKERS")
        )
        
        duration = time.time() - start_
        
        if self.device == 0:
            print(f"Loaded {split} dataset in :: {str(datetime.timedelta(seconds = duration))}")
        
        return loader
    
    def getModel():
        """
        Return:
            model: defined model
        """
        return self.model
    
    def train_epoch(self):
        #Set to train
        self.model.train()
        
        start_ = time.time()
        
        losses = {
            "rpn_cls_loss": 0,
            "rpn_bbox_loss": 0,
            "roi_cls_loss": 0,
            "roi_bbox_loss": 0
        }
        total_loss = 0
        batch_no = 0
        
        scalar = torch.cuda.amp.GradScaler()
        
        for images, targets in self.train_loader:
            batch_no += 1
            images = images.to(self.device)
            targets = {k:v.to(self.device) for k,v in targets.items()}
            
            with torch.cuda.amp.autocast():
                out_losses, _ = self.model(images, targets)
                
                sum_loss = sum(loss for loss in out_losses.values())
            
            for loss_key, loss in out_losses.items():
                losses[loss_key] += float(loss)
            
            
            scalar.scale(sum_loss).backward()
            scalar.step(self.optimizer)
            scalar.update()
            
            total_loss += float(sum_loss)
        
        total_loss = total_loss / batch_no
        losses = {k:(v/batch_no) for k,v in losses.items()}
        duration = time.time() - start_
        
        return total_loss, losses, duration
    
    def evalEpoch(self, epoch):
        eval_metrics = {"map":-1}
        if epoch % self.settings.get("VAL_EPOCH") == 0 and self.with_val:
            eval_start = time.time()
            eval_metrics = evaluate(self.model,self.val_loader,device=self.device)
            eval_duration = time.time() - eval_start
            
            print(f"Evaluated in: {str(datetime.timedelta(seconds = eval_duration))}", flush=True)
        return eval_metrics
    
    def train(self):
        start_ = time.time()
        for epoch in range(self.settings.get("EPOCHS")):
            
            total_loss, losses, duration = self.train_epoch()
            
            self.scheduler.step()
            
            eval_metrics = self.evalEpoch(epoch+1)
            
            if self.device == 0:
                self.logger.update(total_loss, epoch+1, duration, self.model, self.optimizer, eval_metrics, losses)
                self.logger.summarize()
            
        
        duration = time.time() - start_
        
        print(f"Model Finished Training: {str(datetime.timedelta(seconds = duration))}", flush=True)
            
                
            
            
        
