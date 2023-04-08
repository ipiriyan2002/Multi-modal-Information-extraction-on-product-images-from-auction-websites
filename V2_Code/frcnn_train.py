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
from custom.Loaders.base_dataset import BaseDataset
from Utils.utils import load_config_file
#Import other packages
import numpy as np
import os, io, time, datetime, argparse

#=====Utility Functions=====
def parse_args():
    parser = argparse.ArgumentParser(description='Download Pascal 2007 Dataset')
    parser.add_argument("config", help="config file")
    args = parser.parse_args()
    
    return args

def setupDDP(rank, worldsize):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "28961"
    
    init_process_group(backend="nccl", rank=rank, world_size=worldsize)


def runDDPTraining(rank, worldsize, config_name):
    setupDDP(rank, worldsize)
    main(rank, config_name=config_name)
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
    batch_no = 0
    for images, targets in loader:
        #Data, setting to same device
        batch_start = time.time()
        optimizer.zero_grad()
        images = [img.to(device) for img in images]
        targets = [{k:v.to(device) for k,v in target.items()} for target in targets]
        
        with torch.autocast():
            outputs = model(images, targets)
            losses = sum([loss for loss in outputs.values()])
        
        losses.backward()
        optimizer.step()
        batch_duration = time.time() - batch_start
        if (batch_no + 1) % 2 == 0: 
            print(f"(Batch:{batch_no}, Duration:{str(datetime.timedelta(seconds = batch_duration))}, Learning rate:{optimzer.param_groups[0]['lr']}) ==> Loss {losses}", flush=True)

"""
def evaluate(model, loader, device):
    with torch.no_grad():
        model.eval()
        
        for images, targets in loader:
            images = [img.to(device) for img in images]
"""      
    
#=====Main Function=====
def main(device, config_name):
    config_file = load_config_file(config_name)
    
    try:
        os.makedirs(config_file['SAVE_PATH_CHECKPOINT'])
        os.makedirs(config_file['SAVE_PATH_BEST'])
    except:
        print("Warning: File Already Exits! Continuing to next step...")
    
    
    #Model
    model = getFasterRCNN((config_file['ANCHOR_SCALES'],), (config_file['ANCHOR_RATIOS'],), config_file['NUM_CLASSES'])
    model.to(device)
    
    #Dataset
    load_time = time.time()
    
    train_dataset = BaseDataset(config_name, pad=False, split='train')
    #val_dataset = BaseDataset(config_name, pad=False, split='validation')
    
    #Data loaders
    collate = lambda batch: list(zip(*batch))
    train_loader = tu.data.DataLoader(train_dataset, batch_size=config_file['BATCH'], shuffle=True, num_workers=4,collate_fn=collate)
    #val_loader = tu.data.DataLoader(val_dataset, batch_size=config_file['VAL_BATCH'], shuffle=False, num_workers=4,collate_fn=collate)
    
    duration = time.time() - load_time
    print(f"Dataset Loading Time: {str(datetime.timedelta(seconds = duration))}", flush=True)
    
    #Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config_file['L_RATE'], momentum=config_file['MOMENTUM'], weight_decay=config_file['W_DECAY'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config_file['STEP_SIZE'], gamma=config_file['GAMMA'])
    
    for epoch in range(config_file['EPOCHS']):
        epoch_start_time = time.time()
        print(f"===============Epoch {epoch+1}===============", flush=True)
        #Train for one epoch
        train_epoch(model, optimizer, train_loader, device)
        #Reduce learning rate
        scheduler.step()
        #Evaluate on validation dataset
        epoch_duration = time.time() - epoch_start_time
        
        #Saving model every 10 epochs
        if (epoch+1) % 10 == 0:
            save_dict = {
                'epoch': epoch+1,
                'model_dict': model.module.state_dict(),
                'optimizer_dict': optimizer.state_dict() 
            }
            print("Saving Model")
            torch.save(save_dict, self.config['SAVE_PATH_CHECKPOINT'] + f"checkpoint_{epoch+1}.pt")
            print("Saved Model")
        
        print(f"Epoch Duration {str(datetime.timedelta(seconds = epoch_duration))}")
        print(f"===============Epoch Finished===============", flush=True)
    


if __name__ == "__main__":
    args = parse_args()
    config_name = args.config
    worldsize = torch.cuda.device_count()
    mp.spawn(runDDPTraining, args=(worldsize, config_name, args), nprocs=worldsize)
