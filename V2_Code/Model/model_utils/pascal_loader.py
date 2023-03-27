import os, random
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torchvision import ops
import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torch.utils as tu

def getClassDicts():
    cls2idx = {'background': 0, 'person' : 1, 'diningtable' : 2, 'bottle' : 3, 'bird' : 4,
          'sofa' : 5, 'tvmonitor' : 6, 'bicycle' : 7, 'car' : 8, 'bus' : 9,
          'chair' : 10, 'dog' : 11, 'motorbike' : 12, 'cat' : 13, 'aeroplane' : 14, 'horse' : 15,
          'sheep' : 16, 'train' : 17, 'boat' : 18, 'pottedplant' : 19, 'cow' : 20}

    idx2cls = {k : v for v,k in cls2idx.items()}
    
    return cls2idx, idx2cls

def read_voc_xml(xmlFile):
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
    outDict['size'] = (width, height, depth)
    
    #Parse to find all the objects in the annotated image
    objects = parsedXML.findall('object')
    outObjs = []
    for obj in objects:
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
    
    def __init__(self, img_path, ann_path, cls_dict, target_size):
        self.img_path = img_path 
        self.ann_path = ann_path
        self.target_width, self.target_height, self.target_depth = target_size
        
        self.cls_dict = cls_dict 
        
        self.gt_imgs, self.gt_bboxes, self.gt_classes = self.getDataset()
    
    def __len__(self):
        return len(self.gt_imgs)
    
    def __getitem__(self, idx):
        return self.gt_imgs[idx], self.gt_bboxes[idx], self.gt_classes[idx]
    
    def pairList(self):
        pairs = []
        
        for image in os.listdir(self.img_path):
            xml_name = image.split('.')[0] + '.xml'
            ann_name = self.ann_path + xml_name
            
            image_name = self.img_path + image
            
            pairs.append((image_name, ann_name))
        
        return pairs
    
    def normaliseToTarget(self, box, width, height):
        return [
            int(box[0] / (width/self.target_width)),
            int(box[1] / (height/self.target_height)),
            int(box[2] / (width/self.target_width)),
            int(box[3] / (height/self.target_height))
        ]
    
    def getDataset(self):
        pairs = self.pairList()
        
        gt_imgs = []
        gt_bboxes = []
        gt_classes = []
        
        for img_path, ann_path in pairs:
            img = Image.open(img_path)
            img = img.resize((self.target_width, self.target_height))
            
            img_arr = np.asarray(img, dtype='float32') / 255.0
            img_arr = img_arr.transpose(-1,0,1)
            
            gt_imgs.append(img_arr)
            
            annDict = read_voc_xml(ann_path)
            
            og_width, og_height, og_depth = annDict['size']
            
            objs = annDict['objects']
            
            classes = []
            bboxes = []
            
            for obj in objs:
                classes.append(self.cls_dict[obj['class']])
                bbox = obj['bbox']
                
                bboxes.append(self.normaliseToTarget(bbox, og_width, og_height))
            
            
            gt_bboxes.append(torch.tensor(bboxes, dtype=torch.float32))
            gt_classes.append(torch.tensor(classes, dtype=torch.float32))
        
        #Padding the bboxes and classes with the value of 0 which is the numerical value for the background class
        gt_bboxes_padded = pad_sequence(gt_bboxes, batch_first=True, padding_value=0)
        gt_classes_padded = pad_sequence(gt_classes, batch_first=True, padding_value=0)
            
        
        
        return gt_imgs, gt_bboxes_padded, gt_classes_padded
