#Import torch packages
import torch
import torch.utils as tu
import torch.nn as nn
from torchvision import ops, transforms
from torch.nn.utils.rnn import pad_sequence
#Import custom packages
import lib.Loaders.loader_utils as lu
#Import other packages
import os, random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


def getClassDicts():
    """
    Return:
        cls2idx : dictionary of classes to their numerical value
        idx2cls : dictionary of numerical value to their classes
    """
    cls2idx = {'background': 0, 'person' : 1, 'diningtable' : 2, 'bottle' : 3, 'bird' : 4,
          'sofa' : 5, 'tvmonitor' : 6, 'bicycle' : 7, 'car' : 8, 'bus' : 9,
          'chair' : 10, 'dog' : 11, 'motorbike' : 12, 'cat' : 13, 'aeroplane' : 14, 'horse' : 15,
          'sheep' : 16, 'train' : 17, 'boat' : 18, 'pottedplant' : 19, 'cow' : 20}

    idx2cls = {k : v for v,k in cls2idx.items()}
    
    return cls2idx, idx2cls

class VOCDetDataset(tu.data.Dataset):
    
    def __init__(self, img_path, ann_path, cls_dict, target_size, pad=True):
        """
        Arguments:
            img_path (string) : path to images
            ann_path (string) : path to annotations
            cls_dict (Dict) : dictionary containing classes to their numerical value
            target_size (Height x Width x Depth) : target size to resize images
        """
        self.img_path = img_path 
        self.ann_path = ann_path
        self.target_height, self.target_width, self.target_depth = target_size
        
        self.cls_dict = cls_dict 
        self.pad = pad
        self.gt_imgs, self.gt_targets = self.getDataset()
    
    def __len__(self):
        """
        Return:
            size of dataset
        """
        return len(self.gt_imgs)
    
    def __getitem__(self, idx):
        """
        Arguments:
            idx: index to retreive
        Return:
            imgs : image array
            targets : target dictionary
        """
        return self.gt_imgs[idx], self.gt_targets[idx]
    
    
    def getDataset(self):
        """
        Return:
            gt_imgs (list) : list of tensors of shape (Depth x Height x Width)
            gt_bboxes_padded (list) : list of tensors of shape (N, 4) | bounding boxes
            gt_classes_padded (list) : list of tensors of shape (N, 1) | classes
        """
        #Get the paired list of image file and annotation file path
        pairs = lu.pairXmlList(self.img_path, self.ann_path)
        
        gt_imgs = []
        gt_bboxes = []
        gt_classes = []
        
        for img_path, ann_path in pairs:
            #Image processing
            img = Image.open(img_path)
            img = img.resize((self.target_height, self.target_width))
            img_arr = np.asarray(img, dtype='float32') / 255.0
            img_arr = img_arr.transpose(-1,0,1)
            gt_imgs.append(torch.from_numpy(img_arr))
            
            #label processing
            annDict = lu.read_voc_xml(ann_path)
            #Get the original shape
            og_height, og_width, og_depth = annDict['size']
            #Get all objects
            objs = annDict['objects']
            
            classes = []
            bboxes = []
            
            for obj in objs:
                
                bbox = obj['bbox']
                norm_box = lu.normaliseToTarget(bbox, (og_height, og_width), (self.target_heigth, self.target_width))
                if lu.isBox(norm_box):
                    bboxes.append(norm_box)
                    classes.append(self.cls_dict[obj['class']])
            
            #Append tensor form of boxes and classes
            gt_bboxes.append(torch.tensor(bboxes, dtype=torch.float32))
            gt_classes.append(torch.tensor(classes, dtype=torch.int64))
        
        #Padding the bboxes and classes with the value of -1 which is the numerical value for ignore
        if self.pad:
            gt_bboxes = pad_sequence(gt_bboxes, batch_first=True, padding_value=-1)
            gt_classes = pad_sequence(gt_classes, batch_first=True, padding_value=-1)
            
        #Putting all the data into dictionary format
        gt_targets = lu.packTargets(gt_bboxes, gt_classes)
        
        return gt_imgs, gt_targets
