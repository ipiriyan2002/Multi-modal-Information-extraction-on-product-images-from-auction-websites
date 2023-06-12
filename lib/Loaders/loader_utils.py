#Import torch packages
import torch
import torchvision
#Import other packages
import numpy as np
import xml.etree.ElementTree as ET
import os
import random

def normaliseToTarget(box, og_size, target_size):
        """
        Normalise the bounding box to target size
        Return:
            normalised box: [x_min, y_min, x_max, y_max] such that all elements are normalised to target size
        """
        
        height, width = og_size
        target_height, target_width = target_size

        width_ratio, height_ratio = target_width/width, target_height/height
        
        #Unnormalise to target_size
        box[0] *= width_ratio
        box[1] *= height_ratio
        box[2] *= width_ratio
        box[3] *= height_ratio
        
        return box
    

def isBox(box):
    hasArea = ((box[3]-box[1]) * (box[2]-box[0])) > 0
    hasHW = ((box[3]-box[1]) > 0) and ((box[2]-box[0]) > 0)
    return hasArea and hasHW

def packTargets(gt_bboxes, gt_classes, gt_texts=None):
    
    gt_targets = []
    
    for index, bboxes in enumerate(gt_bboxes):
        gt_target = {}
        gt_target['boxes'] = bboxes
        gt_target['labels'] = gt_classes[index]
        gt_target['image_id'] = torch.tensor([index])
        gt_target['area'] = (bboxes[...,2] - bboxes[...,0]) * (bboxes[...,3] - bboxes[...,1])
        gt_target['iscrowd'] = (gt_classes[index] < 0).type(torch.int64)

        if not(gt_texts == None):
            gt_target["text"] = gt_texts[index]


        gt_targets.append(gt_target)
    
    return gt_targets


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

def pairXmlList(img_path, ann_path):
        """
        Return:
            pairs (list): a list of pairs (image file name, annotation file name)
        """
        pairs = []
        #read through the image folder
        for image in os.listdir(img_path):
            #name of the xml annotation file for the image
            xml_name = image.split('.')[0] + '.xml'
            #Add the paths to the image and annotation file name
            ann_name = ann_path + xml_name
            image_name = img_path + image
            
            pairs.append((image_name, ann_name))
        
        return pairs


class DataTransformer(object):
    
    available_transforms = [
        'totensor', 'horizontalflip', 'verticalflip'
    ]
    
    
    def __init__(self, transforms, prob=0.5):
        
        for transform in transforms:
            assert (transform.lower() in DataTransformer.available_transforms), f"{transform} not currently available"
            
        self.transforms = transforms
        self.prob = prob
    
    def toTensor(self, image, target):
        if isinstance(image, torch.Tensor):
            return image, target
        return torchvision.transforms.functional.to_tensor(image), target
    
    def horizontalFlip(self, image, target):
        if random.random() < self.prob:
            image_width = image.shape[-1]
            image = image.flip(-1)
            
            if target != None:
                target['boxes'][...,[0,2]] = image_width - target['boxes'][...,[2,0]]
            
        return image, target
    
    def verticalFlip(self, image, target):
        if random.random() < self.prob:
            image_height = image.shape[-2]
            image = image.flip(-2)
            
            if target != None:
                target['boxes'][...,[1,3]] = image_height - target['boxes'][...,[3,1]]
        return image, target
    
    def applyTransform(self, image, target, transform):
        
        if transform.lower() == 'totensor':
            return self.toTensor(image, target)
        elif transform.lower() == 'horizontalflip':
            return self.horizontalFlip(image, target)
        elif transform.lower() == 'verticalflip':
            return self.verticalFlip(image, target)
       
        return image, target
    
    
    def __call__(self, image, target):
        
        for transform in self.transforms:
            image, target = self.applyTransform(image, target, transform)
        
        return image, target
        