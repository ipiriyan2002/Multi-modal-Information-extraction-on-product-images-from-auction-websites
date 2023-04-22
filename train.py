#Import torch packages
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
#Import custom packages
from lib.Model.faster_rcnn_detector import FRCNNDetector
from lib.Loaders.base_dataset import BaseDataset
import Utils.utils as utils
from Utils.eval import evaluate
from Utils.config_loader import ConfigLoader
from Utils.logger_utils import TrainingLogger
from Utils.train_utils import *
#Import other packages
import os, io, time, datetime
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train custom model')
    parser.add_argument("config", help="config file")
    parser.add_argument("--socket", default="28960", help="master_port")
    #parser.add_argument("--resume", action='store_true', default=False, help="resume from checkpoint")
    parser.add_argument("--resume-path", default="", help="checkpoint path")
    args = parser.parse_args()
    
    return args

def setupDDP(rank, worldsize, port_socket):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_socket
    
    init_process_group(backend="nccl", rank=rank, world_size=worldsize)

def main(rank, settings, world_size, resume_path):
    settings = ConfigLoader(settings)
    resume_point = 0
    prev_min_loss = 1e15
    
    #Defining model
    model = FRCNNDetector(settings)
    model = model.to(rank)
        
    #Defining Optimizer
    if settings.get("OPT_TYPE") == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=settings.get["L_RATE"], 
            weight_decay=settings.get['WEIGHT_DECAY']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=settings.get('L_RATE'), 
            momentum=settings.get('MOMENTUM'), 
            weight_decay=settings.get('W_DECAY')
        )
    #Define Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=settings.get('STEP_SIZE'), 
        gamma=settings.get('GAMMA')
    )
    
    #Defining the Model in terms of DistributedDataParallel
    if settings.get("DDP"):
        model = DDP(
            model, device_ids=[rank], find_unused_parameters=True
        )
    
    if resume == "":
        resume_dict = torch.load(resume_path)
        resume_point = resume_dict["epoch"]
        prev_min_loss = resume_dict["loss"]
        model.load_state_dict(resume_dict['model_dict'])
        optimizer.load_state_dict(resume_dict['optimizer_dict'])
    
    logger = TrainingLogger(settings, world_size)
    if device == 0:
        logger.init_print_settings()
    
    #Defining training and validation loader
    train_loader = getDataLoader(rank, settings, split="train")
    val_loader = getDataLoader(rank, settings, split="validation")
    
    
    start_ = time.time()
    
    
    for epoch in range(resume_point, settings.get("EPOCHS")):
        total_losses, losses, duration = train_epoch(model, optimizer, train_loader, rank)
        
        scheduler.step()
        
        eval_metrics = evalEpoch(epoch+1, model, val_loader, rank)
        
        if device == 0:
            logger.update(total_loss, epoch+1, duration, model, optimizer, eval_metrics, losses)
            logger.summarize()

    duration = time.time() - start_
    
    print(f"Model Finished Training: {str(datetime.timedelta(seconds = duration))}", flush=True)
    
    

def runDDPTraining(rank, worldsize, config_name, args):
    setupDDP(rank, worldsize, args.socket)
    main(rank, settings=config_name, world_size=worldsize, resume_path=args.resume_path)
    destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    config_name = args.config
    worldsize = torch.cuda.device_count()
    mp.spawn(runDDPTraining, args=(worldsize, config_name, args), nprocs=worldsize)
