#Import torch packages
import torch
import torch.utils as tu
#Import custom packages
from Utils.utils import load_config_file
from Data_Loaders.cordv2.cord_loader import CordDataset
from Data_Loaders.voc2007.pascal_loader import VOCDetDataset, getClassDicts
#Import other packages
import numpy as np
import os,random


class BaseDataset:
    def __init__(self, config_file, base_config='BASE_PATHS.yaml', pad=True, split=""):
        """
        Arguments:
            config_file (string) : load the config file for the dataset
            base_config (string) : load the base config file with the paths to the downloaded datasets
        """
        self.config = load_config_file(config_file)
        self.base_config = load_config_file(base_config)
        self.pad = pad
        self.dataset = self.config['DATASET']
        self.img_width, self.img_height, self.img_depth = self.config["IMG_WIDTH"], self.config["IMG_HEIGHT"], self.config["CHANNELS"]
        
        self.split = self.config['SPLIT'] if split == "" else split
        
    
    def getDataset(self):
        """
        Return:
            dataset (torch dataset): dataset defined in the config file
        """
        if self.dataset.lower() == 'cordv2':
            #Return cord dataset
            return CordDataset(self.base_config['CORD_V2_PATH'],
                              target_size=(self.img_height, self.img_width, self.img_depth), split=self.split, pad=self.pad)
        elif self.dataset.lower() == 'voc2007':
            #Return the pascal voc 2007 dataset
            #Get the dataset paths from the base config file with respect to split
            img_path = self.base_config['VOC2007_TRAIN_IMG_PATH'] if (self.split.lower() == 'train') else self.base_config['VOC2007_TEST_IMG_PATH']
            ann_path = self.base_config['VOC2007_TRAIN_ANN_PATH'] if (self.split.lower() == 'train') else self.base_config['VOC2007_TEST_ANN_PATH']
            cls2idx, _ = getClassDicts()
            
            return VOCDetDataset(img_path, ann_path, cls_dict=cls2idx, 
                                 target_size=(self.img_height, self.img_width, self.img_depth))