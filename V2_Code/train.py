from Model.one_stage_detector import OneStageDetector
import os
import numpy as np
import pandas as pd
import torch
import io
import utils
import preprocess as pre
import data_augmentation as aug
import DataLoader as dl
import multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

class OSDTrainer():
    def __init__(self, rank, config_name=None):
        assert config_path != None, "YAML Configuration File Name Under Configs Folder Is Needed"
        
        self.config = utils.load_config_file(config_name)
        self.rank = rank
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Defining training dataset and model
        self.train_dataset = dl.getCordTorchDatasetLoader(config_name, split="train", include_sampler=True)
        self.model = OneStageDetector(self.config["IMG_WIDTH"], self.config["IMG_HEIGHT"], 512, 16,device=self.rank)
        self.model = self.model.to(self.rank)
        
        #Defining Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["L_RATE"])
        
        #Defining the Model in terms of DistributedDataParallel
        self.model = DDP(self.model, device_ids=[self.rank])
        
        #Defining all losses list to monitor per epoch loss
        self.all_losses = []
    
    def getAllLosses(self):
        return self.all_losses
    
    def printOutput(self, eCount, eLoss):
        meanLoss = sum(self.all_losses) / len(self.all_losses)
        print("(GPU {0}, Epoch {1}) ==> Mean Loss :: {2}  | Epoch Loss :: {3}".format(self.device.index,eCount, meanLoss, eLoss))
    
    
    def single_epoch_run(self, epoch_count, images, bboxes, conf_scores):
        self.optimizer.zero_grad()
        
        model_loss = self.model(images, bboxes, conf_scores)
        
        model_loss.backward()
        
        self.optimizer.step()
        
        self.all_losses.append(model_loss)
        
        self.printOutput(epoch_count, model_loss)
        
        return model_loss
    
    def trainModel(self):
        save_path = "./Saved_Models/OSD_E_{0}_B_{1}_{2}x{3}_Checkpoint.pt".format(
                        self.config['EPOCHS'],self.config['BATCH'], self.config['IMG_WIDTH'], self.config['IMG_HEIGHT'])
        for epoch in range(self.config['EPOCHS']):
            prev_min_loss = 1e6
            for imgs, targets in self.train_dataset:
                imgs = imgs.to(self.rank)
                bboxes = target['bboxes'].to(self.rank)
                conf_scores = target['confidence scores'].to(self.rank)
                
                eLoss = self.single_epoch_run(epoch, imgs, bboxes, conf_scores)
                
                if (self.rank == 0) and (eLoss <= prev_min_loss):
                    model_dict = self.model.module.state_dict()
                    torch.save(model_dict, save_path)
                    prev_min_loss = eLoss
            

def setupDDP(rank, worldsize):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "28961"
    
    init_process_group(backend="nccl", rank=rank, worldsize=worldsize)

def runDDPTraining(rank, worldsize, config_name):
    setupDDP(rank, worldsize)
    modelTrainer = OSDTrainer(rank, config_name)
    modelTrainer.trainModel()
    destroy_process_group()

if __name__ == "__main__":
    config_name = "params_cord_initial.yaml"
    worldsize = torch.cuda.device_count()
    mp.spawn(runDDPTraining, args=(worldsize, config_name), nprocs=worldsize)
