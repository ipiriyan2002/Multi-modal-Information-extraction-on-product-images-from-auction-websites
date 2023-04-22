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

def getDataLoader(device, settings, split="train"):
    start_ = time.time()
    
    #Get dataset
    dataset = BaseDataset(settings, pad=settings.get("PAD"), split=split)
    #Get loader
    loader = DataLoader(
        dataset.getDataset(), batch_size=settings.get("BATCH"),
        shuffle=(split=="train"), pin_memory=True,
        num_workers=settings.get("NUM_WORKERS")
    )
    duration = time.time() - start_
        
    if device == 0:
        print(f"Loaded {split} dataset in :: {str(datetime.timedelta(seconds = duration))}")
        
    return loader

def train_epoch(model, optimizer, train_loader, device):
    #Set to train
    model.train()
    
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
        
    for images, targets in train_loader:
        optimizer.zero_grad()
        batch_no += 1
        images = images.to(device)
        targets = {k:v.to(device) for k,v in targets.items()}
            
        with torch.cuda.amp.autocast():
            out_losses, _ = model(images, targets)
                
            sum_loss = sum(loss for loss in out_losses.values())
            
        for loss_key, loss in out_losses.items():
            losses[loss_key] += float(loss)
            
            
        scalar.scale(sum_loss).backward()
        scalar.step(optimizer)
        scalar.update()
            
        total_loss += float(sum_loss)
        
    total_loss = total_loss / batch_no
    losses = {k:(v/batch_no) for k,v in losses.items()}
    duration = time.time() - start_
        
    return total_loss, losses, duration

def evalEpoch(epoch, model, val_loader, device):
    eval_metrics = {"map":-1}
    if epoch % self.settings.get("VAL_EPOCH") == 0:
        eval_start = time.time()
        eval_metrics = evaluate(model,val_loader,custom=True,device=device)
        eval_duration = time.time() - eval_start
            
    print(f"Evaluated in: {str(datetime.timedelta(seconds = eval_duration))}", flush=True)

    return eval_metrics
    
            
                
            
            
        
