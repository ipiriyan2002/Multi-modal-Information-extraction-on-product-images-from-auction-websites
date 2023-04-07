#Import torch packages
import torch
import torch.utils as tu
import torch.nn as nn
from torchvision import ops, transforms
from torch.nn.utils.rnn import pad_sequence
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


def read_voc_xml(xmlFile):
    """
    Arguments:
        xmlFile: the path to the xml file
    Return:
        outDict: dictionary containing the meta and training info
    """
    outDict = {}
    
    #Parse the xml file
    parsedXML = ET.parse(xmlFile)
    
    #Get the name of the image file
    imgName = parsedXML.find('filename').text
    outDict['img_name'] = imgName
    
    #Get the original size of the image and save in form (width, height, depth)
    size = parsedXML.find('size')
    width = eval(size.find('width').text)
    height = eval(size.find('height').text)
    depth = eval(size.find('depth').text)
    outDict['size'] = (height, width, depth)
    
    #Parse to find all the objects in the annotated image
    objects = parsedXML.findall('object')
    outObjs = []
    for obj in objects:
        #Class dictionary containing the class name, bounding box
        classDict = {}
        
        class_name = obj.find('name').text
        classDict['class'] = class_name
        
        bbox = obj.find('bndbox')
        x1 = eval(bbox.find('xmin').text)
        y1 = eval(bbox.find('ymin').text)
        x3 = eval(bbox.find('xmax').text)
        y3 = eval(bbox.find('ymax').text)
        
        classDict['bbox'] = [x1,y1,x3,y3]
        
        outObjs.append(classDict)
    
    outDict['objects'] = outObjs
    
    return outDict


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
    
    def pairList(self):
        """
        Return:
            pairs (list): a list of pairs (image file name, annotation file name)
        """
        pairs = []
        #read through the image folder
        for image in os.listdir(self.img_path):
            #name of the xml annotation file for the image
            xml_name = image.split('.')[0] + '.xml'
            #Add the paths to the image and annotation file name
            ann_name = self.ann_path + xml_name
            image_name = self.img_path + image
            
            pairs.append((image_name, ann_name))
        
        return pairs
    
    def normaliseToTarget(self, box, width, height):
        """
        Normalise the bounding box to target size
        Return:
            normalised box: [x_min, y_min, x_max, y_max] such that all elements are normalised to target size
        """
        return [
            int(box[0] / (width/self.target_width)),
            int(box[1] / (height/self.target_height)),
            int(box[2] / (width/self.target_width)),
            int(box[3] / (height/self.target_height))
        ]
    
    def getDataset(self):
        """
        Return:
            gt_imgs (list) : list of tensors of shape (Depth x Height x Width)
            gt_bboxes_padded (list) : list of tensors of shape (N, 4) | bounding boxes
            gt_classes_padded (list) : list of tensors of shape (N, 1) | classes
        """
        #Get the paired list of image file and annotation file path
        pairs = self.pairList()
        
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
            annDict = read_voc_xml(ann_path)
            #Get the original shape
            og_height, og_width, og_depth = annDict['size']
            #Get all objects
            objs = annDict['objects']
            
            classes = []
            bboxes = []
            
            for obj in objs:
                classes.append(self.cls_dict[obj['class']])
                bbox = obj['bbox']
                
                bboxes.append(self.normaliseToTarget(bbox, og_width, og_height))
            
            #Append tensor form of boxes and classes
            gt_bboxes.append(torch.tensor(bboxes, dtype=torch.float32))
            gt_classes.append(torch.tensor(classes, dtype=torch.int64))
        
        #Padding the bboxes and classes with the value of -1 which is the numerical value for ignore
        if self.pad:
            gt_bboxes = pad_sequence(gt_bboxes, batch_first=True, padding_value=-1)
            gt_classes = pad_sequence(gt_classes, batch_first=True, padding_value=-1)
            
        #Putting all the data into dictionary format
        gt_targets = []
        
        for index, bboxes in enumerate(gt_bboxes):
            gt_target = {}
            gt_target['boxes'] = bboxes
            gt_target['labels'] = gt_classes[index]
            gt_target['image_id'] = torch.tensor([index])
            gt_target['area'] = (bboxes[...,2] - bboxes[...,0]) * (bboxes[...,3] - bboxes[...,1])
            gt_target['iscrowd'] = (gt_classes[index] < 0).type(torch.int64)
            gt_targets.append(gt_target)
        
        return gt_imgs, gt_targets
