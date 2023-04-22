import sys
sys.path.append("..")

#Import torch packages
import torch
import torch.utils as tu
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
#Import custom packages
from lib.Loaders.base_dataset import BaseDataset
from Utils.utils import load_config_file
from Utils.eval import evaluate
from Utils.config_loader import ConfigLoader
from Utils.logger_utils import TrainingLogger
#Import other packages
import numpy as np
import os, io, time, datetime, argparse

#=====Utility Functions=====
def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster RCNN pre-implemented in pytorch')
    parser.add_argument("config", help="config file")
    parser.add_argument("--socket", default="28961", help="socket")
    parser.add_argument("--resume-path", default="", help="weights path for resuming")
    args = parser.parse_args()
    
    return args

def setupDDP(rank, worldsize,socket):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = socket
    
    init_process_group(backend="nccl", rank=rank, world_size=worldsize)


def runDDPTraining(rank, worldsize, config_name,args):
    setupDDP(rank, worldsize,args.socket)
    main(rank, worldsize, config_name=config_name, resume_path=args.resume_path)
    destroy_process_group()

#=====Model Functions=====
def getFasterRCNN(aspect_sizes, aspect_ratios, num_classes):
    #Defining backbone
    backbone = torchvision.models.vgg16(weights="DEFAULT").features
    backbone.out_channels = 512
    #Defining Anchor Generator
    anchorGenerator = AnchorGenerator(sizes=aspect_sizes, aspect_ratios=aspect_ratios)
    
    model = FasterRCNN(backbone, rpn_anchor_generator=anchorGenerator, num_classes=num_classes)
    
    return model


def train_epoch(model, optimizer, loader, device):
    model.train()
    epoch_start = time.time()
    batch_no = 0
    temp_losses = 0
    scalar = torch.cuda.amp.GradScaler()
    for images, targets in loader:
        #Data, setting to same device
        optimizer.zero_grad()
        images = [img.to(device) for img in images]
        targets = [{k:v.to(device) for k,v in target.items()} for target in targets]
        
        with torch.cuda.amp.autocast():
            losses = model(images, targets)
            print(losses)
            losses = sum([loss for loss in losses.values()])
        
        temp_losses += float(losses)
        batch_no += 1
        scalar.scale(losses).backward()
        scalar.step(optimizer)
        scalar.update()

    temp_losses = temp_losses / batch_no
    duration = time.time() - epoch_start
    return temp_losses, duration
     
def collate(batch):
    return list(zip(*batch))

#=====Main Function=====
def main(device, worldsize, config_name, resume_path):
    config_file = ConfigLoader(config_name)
    
    #Logger
    logger = TrainingLogger(config_file, worldsize)
    
    if device == 0:
        logger.init_print_settings()
    
    #Model
    model = getFasterRCNN((config_file.get('ANCHOR_SCALES'),), (config_file.get('ANCHOR_RATIOS'),), config_file.get('NUM_CLASSES'))
    model = model.to(device)
    
    #Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config_file.get('L_RATE'), momentum=config_file.get('MOMENTUM'), weight_decay=config_file.get('W_DECAY'))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config_file.get('STEP_SIZE'), gamma=config_file.get('GAMMA'))
    
    if resume_path != "":
        try:
            resume_dict = torch.load(resume_path)
        except:
            raise ValueError("Expected pytorch save file")
        
        start_epoch = resume_dict['epoch']
        model.load_state_dict(resume_dict['model_dict'])
        optimizer.load_state_dict(resume_dict['optimizer_dict'])
    else:
        start_epoch = 0
    
    #Dataset
    load_time = time.time()
    
    train_dataset = BaseDataset(config_file, pad=False, split='train')
    val_dataset = BaseDataset(config_file, pad=False, split='validation')
    
    #Data loaders
    train_loader = tu.data.DataLoader(train_dataset.getDataset(), batch_size=config_file.get('BATCH'), shuffle=True, pin_memory=True, num_workers=0*worldsize, collate_fn=collate)
    
    val_loader = tu.data.DataLoader(val_dataset.getDataset(), batch_size=config_file.get('VAL_BATCH'), shuffle=False, pin_memory=True, num_workers=0*worldsize, collate_fn=collate)
    
    duration = time.time() - load_time
    if device == 0:
        print(f"Dataset Loading Time: {str(datetime.timedelta(seconds = duration))}", flush=True)
    
    total_start = time.time()
    
    for epoch in range(start_epoch, config_file.get('EPOCHS')):
        #Train for one epoch
        losses, duration = train_epoch(model, optimizer, train_loader, device)
            
        #Reduce learning rate
        scheduler.step()
        #Evaluate on validation dataset
        eval_metrics = {"map":-1}
        
        if (epoch+1) % config_file.get("VAL_EPOCH") == 0:
            eval_start = time.time()
            eval_metrics = evaluate(model,val_loader,device=device)
            eval_duration = time.time() - eval_start
            model.train() #Resetting to training mode
        
        if device == 0:
            logger.update(losses, epoch+1, duration, model, optimizer, eval_metrics)
            logger.summarize()
    
    total_duration = time.time() - total_start
    print(f"Model Finished Training: {str(datetime.timedelta(seconds = total_duration))}", flush=True)
    
if __name__ == "__main__":
    #mp.set_start_method("fork")
    args = parse_args()
    config_name = args.config
    worldsize = torch.cuda.device_count()
    mp.spawn(runDDPTraining, args=(worldsize, config_name,args), nprocs=worldsize)
