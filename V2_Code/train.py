import os
import torch
import io
#import preprocess as pre
#import data_augmentation as aug
#import DataLoader as dl
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from OSDTrainer import OSDTrainer

def setupDDP(rank, worldsize):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "28961"
    
    init_process_group(backend="nccl", rank=rank, world_size=worldsize)

def runDDPTraining(rank, worldsize, config_name):
    setupDDP(rank, worldsize)
    modelTrainer = OSDTrainer(rank, config_name=config_name)
    modelTrainer.trainModel()
    destroy_process_group()

if __name__ == "__main__":
    config_name = "VOC_E50_256-256.yaml"
    worldsize = torch.cuda.device_count()
    mp.spawn(runDDPTraining, args=(worldsize, config_name), nprocs=worldsize)
