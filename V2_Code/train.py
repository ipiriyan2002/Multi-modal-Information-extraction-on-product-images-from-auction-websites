import os
import torch
import io
#import preprocess as pre
#import data_augmentation as aug
#import DataLoader as dl
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from OSDTrainer import OSDTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Download Pascal 2007 Dataset')
    parser.add_argument("config", help="config file")
    parser.add_argument("--resume", action='store_true', default=False, help="resume from checkpoint")
    parser.add_argument("--resume-path", help="checkpoint path")
    args = parser.parse_args()
    
    return args

def setupDDP(rank, worldsize):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "28961"
    
    init_process_group(backend="nccl", rank=rank, world_size=worldsize)

def getOSDArgs(args):
    
    resume = args.resume
    resume_path = args.resume_path if (args.resume_path) else ""
    
    return resume, resume_path

def runDDPTraining(rank, worldsize, config_name, args):
    resume, r_path = getOSDArgs(args)
    
    setupDDP(rank, worldsize)
    modelTrainer = OSDTrainer(rank, config_name=config_name, resume=resume, resume_path=r_path)
    modelTrainer.trainModel()
    destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    config_name = args.config
    worldsize = torch.cuda.device_count()
    mp.spawn(runDDPTraining, args=(worldsize, config_name, args), nprocs=worldsize)
