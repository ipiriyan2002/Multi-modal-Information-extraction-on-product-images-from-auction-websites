#Import torch packages
import torch
import torch.utils as tu
#Import custom packages
from Utils.utils import load_config_file
#Import Custom Loaders
from lib.Loaders.cord_loader import CordDataset
from lib.Loaders.pascal_loader import VOCDetDataset, getClassDicts
#Import other packages
import numpy as np
import os,random


class BaseDataset:
    def __init__(self, settings, pad=True, split=""):
        """
        Arguments:
            setting (ConfigLoader) : loaded config class
        """
        self.settings = settings
        self.pad = pad
        self.dataset = self.settings.get('DATASET')
        self.img_width, self.img_height, self.img_depth = self.settings.get("IMG_WIDTH"), self.settings.get("IMG_HEIGHT"), self.settings.get("CHANNELS")
        
        self.split = self.settings.get('SPLIT') if split == "" else split
        
    
    def getDataset(self):
        """
        Return:
            dataset (torch dataset): dataset defined in the config file
        """
        if self.dataset.lower() == 'cordv2':
            #Return cord dataset
            return CordDataset(self.settings.get('CORD_V2_PATH'),
                              target_size=(self.img_height, self.img_width, self.img_depth), split=self.split, pad=self.pad)
        elif self.dataset.lower() == 'voc2007':
            #Return the pascal voc 2007 dataset
            #Get the dataset paths from the base config file with respect to split
            img_path = self.settings.get('VOC2007_TRAIN_IMG_PATH') if (self.split.lower() == 'train') else self.settings.get('VOC2007_TEST_IMG_PATH')
            ann_path = self.settings.get('VOC2007_TRAIN_ANN_PATH') if (self.split.lower() == 'train') else self.settings.get('VOC2007_TEST_ANN_PATH')
            cls2idx, _ = getClassDicts()
            
            return VOCDetDataset(img_path, ann_path, cls_dict=cls2idx, 
                                 target_size=(self.img_height, self.img_width, self.img_depth))
