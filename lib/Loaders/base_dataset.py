#Import torch packages
import torch
import torch.utils as tu
#Import custom packages
from Utils.utils import load_config_file
#Import Custom Loaders
from lib.Loaders.cord_loader import CordDataset
from lib.Loaders.pascal_loader import VOCDetDataset
from lib.Loaders.loader_utils import *
#Import other packages
import numpy as np
import os,random

class BaseDataset:
    def __init__(self, settings, pad=False, split=""):
        """
        Arguments:
            setting (ConfigLoader) : loaded config class
        """
        self.settings = settings
        self.pad = pad

        #dataset name
        self.dataset = self.settings.get('DATASET')

        #dataset features
        self.class_type = self.settings.get("CLASS_TYPE")
        self.split = self.settings.get('SPLIT') if split == "" else split
        self.transforms = DataTransformer(self.settings.get("TRANSFORMS"), prob=0.5)
        

    def getCordDataset(self):
        """
        :return: Cord Dataset
        """

        dataset = CordDataset(
            path=self.settings.get('CORD_V2_PATH'),
            split=self.split,
            class_type=self.class_type,
            pad=self.pad,
            transform=self.transforms
        )

        self.settings.setValue('NUM_CLASSES', dataset.num_classes)

        #Set features gathered from dataset
        #if self.split.lower() == "train":
        #    self.settings.setValue('IMAGE_MEAN', dataset.mean.cpu().tolist())
        #    self.settings.setValue('IMAGE_STD', dataset.std.cpu().tolist())

        return dataset

    def getVoc2007Dataset(self):
        img_path = self.settings.get('VOC2007_TRAIN_IMG_PATH') if (self.split.lower() == 'train') else self.settings.get('VOC2007_TEST_IMG_PATH')
        ann_path = self.settings.get('VOC2007_TRAIN_ANN_PATH') if (self.split.lower() == 'train') else self.settings.get('VOC2007_TEST_ANN_PATH')

        dataset = VOCDetDataset(
            img_path=img_path,
            ann_path=ann_path,
            class_type=self.class_type,
            pad=self.pad,
            transform=self.transforms
        )

        self.settings.setValue('NUM_CLASSES', dataset.num_classes)

        #if self.split.lower() == "train":
        #    self.settings.setValue('IMAGE_MEAN', dataset.mean.cpu().tolist())
        #    self.settings.setValue('IMAGE_STD', dataset.std.cpu().tolist())

        return dataset
        

    def getDataset(self):
        """
        Return:
            dataset (torch dataset): dataset defined in the config file
        """
        if self.dataset.lower() == 'cordv2':
            return self.getCordDataset()
            
        elif self.dataset.lower() == 'voc2007':
            return self.getVoc2007Dataset()
