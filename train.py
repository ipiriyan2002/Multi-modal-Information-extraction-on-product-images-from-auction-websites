import os
import torch
import io
#import preprocess as pre
#import data_augmentation as aug
#import DataLoader as dl
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from Utils.Trainer import Trainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train custom model')
    parser.add_argument("config", help="config file")
    parser.add_argument("--socket", default="28960", help="master_port")
    parser.add_argument("--resume", action='store_true', default=False, help="resume from checkpoint")
    parser.add_argument("--resume-path", default="", help="checkpoint path")
    args = parser.parse_args()
    
    return args

def setupDDP(rank, worldsize, port_socket):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_socket
    
    init_process_group(backend="nccl", rank=rank, world_size=worldsize)


def runDDPTraining(rank, worldsize, config_name, args):
    setupDDP(rank, worldsize, args.socket)
    modelTrainer = Trainer(rank, settings=config_name, with_val=True, world_size=worldsize,resume=args.resume, resume_path=args.resume_path)
    modelTrainer.train()
    destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    config_name = args.config
    worldsize = torch.cuda.device_count()
    mp.spawn(runDDPTraining, args=(worldsize, config_name, args), nprocs=worldsize)
