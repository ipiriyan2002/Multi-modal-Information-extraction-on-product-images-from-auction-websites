#Import torch packages
import torch
#Import other packages
import numpy as np
import xml.etree.ElementTree as ET
import os

def normaliseToTarget(box, og_size, target_size):
        """
        Normalise the bounding box to target size
        Return:
            normalised box: [x_min, y_min, x_max, y_max] such that all elements are normalised to target size
        """
        
        height, width = og_size
        target_height, target_width = target_size
        
        return [
            int(box[0] / (width/target_width)),
            int(box[1] / (height/target_height)),
            int(box[2] / (width/target_width)),
            int(box[3] / (height/target_height))
        ]
    

def isBox(box):
    return ((box[3]-box[1]) > 0) and ((box[2]-box[0]) > 0)


def packTargets(gt_bboxes, gt_classes):
    
    gt_targets = []
    
    for index, bboxes in enumerate(gt_bboxes):
        gt_target = {}
        gt_target['boxes'] = bboxes
        gt_target['labels'] = gt_classes[index]
        gt_target['image_id'] = torch.tensor([index])
        gt_target['area'] = (bboxes[...,2] - bboxes[...,0]) * (bboxes[...,3] - bboxes[...,1])
        gt_target['iscrowd'] = (gt_classes[index] < 0).type(torch.int64)
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