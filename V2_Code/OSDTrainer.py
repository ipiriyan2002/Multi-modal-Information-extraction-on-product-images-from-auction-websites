from Model.one_stage_detector import OneStageDetector
from Model.model_utils.pascal_loader import *
from cord_loader import *
import os
import numpy as np
#import pandas as pd
import torch
import io
import utils
import time
import datetime
#import preprocess as pre
#import data_augmentation as aug
#import DataLoader as dl
from torch.nn.parallel import DistributedDataParallel as DDP

class OSDTrainer():
    def __init__(self, rank, config_name=None, resume=False, resume_path=""):
        assert config_name != None, "YAML Configuration File Name Under Configs Folder Is Needed"
        
        self.config = utils.load_config_file(config_name)
        self.rank = rank
        self.resume_point = 0
        self.prev_min_loss = 1e6
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Defining training dataset and model
        
        #self.cls2idx, self.idx2cls = getClassDicts()
        self.img_width, self.img_height, self.img_depth = self.config["IMG_WIDTH"], self.config["IMG_HEIGHT"], self.config["CHANNELS"]
        
        #self.train_dataset = VOCDetDataset(self.config['PASCAL_TRAIN_IMG_PATH'], self.config['PASCAL_TRAIN_ANN_PATH'],
        #                                               cls_dict=self.cls2idx, 
        #                                               target_size=(self.img_width, self.img_height, self.img_depth))
        
        self.train_dataset = CordDataset(self.config['CORD_PATH'], self.config['CLASSES'], 
                                         target_size=(self.img_width, self.img_height, self.img_depth), split='train')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config['BATCH'], shuffle=True)
        
        self.model = OneStageDetector((self.img_depth, self.img_width, self.img_height),
                 conf_score_weight = self.config['CONF_LOSS_WEIGHT'], bbox_weight = self.config['BBOX_LOSS_WEIGHT'],
                 pos_anchor_thresh = self.config['POS_ANCHOR_THRESH'], neg_anchor_thresh = self.config['NEG_ANCHOR_THRESH'], anc_ratio=self.config['ANCHOR_RATIO'], 
                 anchor_scales=self.config['ANCHOR_SCALES'], anchor_ratios = self.config['ANCHOR_RATIOS'], stride=self.config['STRIDE'],device=self.rank)
        
        #Defining Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["L_RATE"])
        
        #Defining the Model in terms of DistributedDataParallel
        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
        
        
        if resume:
            self.resume_dict = torch.load(resume_path)
            self.model.load_state_dict(resume_dict['model_dict'])
            self.optimizer.load_state_dict(resume_dict['optimizer_dict'])
            self.resume_point = self.resume_dict['epoch']
            self.prev_min_loss = self.resume_dict['loss']
        
        try:
            os.makedirs(self.config['SAVE_PATH_CHECKPOINT'])
            os.makedirs(self.config['SAVE_PATH_BEST'])
        except:
            print("Warning: File Already Exits! Continuing to next step...")
        
    
    def getModel():
        return self.model
    
    def printOutput(self, eCount, eLoss, time):

        print("(GPU {0}, Epoch {1}, Duration(s) {2}) ==> Mean Loss :: {3}".format(self.rank,eCount, str(datetime.timedelta(seconds = time)), eLoss), flush=True)
    
    
    def single_batch_run(self, images, bboxes, conf_scores):
        self.optimizer.zero_grad()
        
        model_loss = self.model(images, bboxes, conf_scores)
        
        model_loss.backward()
        
        self.optimizer.step()
        
        return model_loss
    
    def checkpointSaver(self, loss, prev_loss, epoch, final=False, duration=0):
        if self.rank == 0:
            save_dict = {
                    "epoch" : epoch +1,
                    "loss" : loss,
                    "model_dict" : self.model.module.state_dict(),
                    "optimizer_dict" : self.optimizer.state_dict()
                }
            
            #Saving final model
            if final:
                print("Saving Final Model ==> Trained Epochs : {0} | Final Loss : {1} | Training Duration : {2}".format(
                    epoch+1, loss, str(datetime.timedelta(seconds=duration))
                ), flush=True)
                
                torch.save(save_dict, self.config['SAVE_PATH_CHECKPOINT'] + f"last_checkpoint.pt")
                
                print("Model Saved", flush=True)
                return 0
            #Saving Checkpoint
            if ((epoch+1) % self.config['SAVE_ENTRY']) == 0:
                print("Saving Model At Epoch {0}...".format(epoch+1), flush=True)
                torch.save(save_dict, self.config['SAVE_PATH_CHECKPOINT'] + f"checkpoint_{epoch+1}.pt")
                print("Model Saved", flush=True)
            #Saving best model
            if (loss < prev_loss):
                print("Saving Best Model", flush=True)
                torch.save(save_dict, self.config['SAVE_PATH_BEST'] + "best_model.pt")
                print("Best Model Saved", flush=True)
                return loss
        
        return prev_loss
            
    
    def trainModel(self):
        self.model.train()
        model_train_time_start = time.time()
        model_final_loss = 0
        
        for epoch in range(self.resume_point, self.config['EPOCHS']):
            
            if self.rank == 0:
                print(f"=====(Epoch {epoch+1})=====", flush=True)
            #Starting epoch timer
            start_time = time.time()
            #Storing epcoh total loss without storing history
            epoch_total_loss = 0.0
            num_batches = 0.0
            #One epoch run
            for imgs, bboxes, classes in self.train_loader:
                #Starting timer for each batch
                batch_timer = time.time()
                #Transferring data to same device
                imgs = imgs.to(self.rank)
                bboxes = bboxes.to(self.rank)
                classes = classes.to(self.rank)
                
                #Get loss for one parse
                eLoss = self.single_batch_run(imgs, bboxes, classes)
                epoch_total_loss += float(eLoss)
                num_batches += float(1)
                
                if (self.rank == 0) and ((num_batches - 1) % 2 == 0):
                    print(f"Batch {num_batches - 1} ==> Batch Loss :: {eLoss} | Duration :: {str(datetime.timedelta(seconds=time.time() - batch_timer))}", flush=True)
            
            epoch_total_loss /= num_batches
            
            #Ending epoch timer
            self.printOutput(epoch, epoch_total_loss, time.time() - start_time)
            
            self.prev_min_loss = self.checkpointSaver(epoch_total_loss, self.prev_min_loss, epoch)
            model_final_loss = epoch_total_loss
                
        #Finished Training
        self.checkpointSaver(model_final_loss, 0, self.config['EPOCHS'], final=True, duration=time.time() - model_train_time_start)
        print("Finished Training", flush=True)

