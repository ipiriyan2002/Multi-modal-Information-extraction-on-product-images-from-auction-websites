import os, random
import numpy as np
import pandas as pd
import torch
from torchvision import ops
from torch.nn.utils.rnn import pad_sequence
import torch.utils as tu
import io
from pillow import Image

class CordDataset(tu.data.Dataset):
    def __init__(self, path, classes, target_size, split):
        assert (split in ['train', 'test', 'validation']) 
        
        self.path = path
        self.target_width, self.target_height, self.target_depth = target_size
        
        self.classes = classes
        
        self.cls_dict = self.getClassDict()
        
        self.split = split
        self.splitDict = {
            'train' : ["train-00000-of-00004-b4aaeceff1d90ecb.parquet", "train-00001-of-00004-7dbbe248962764c5.parquet",
                       "train-00002-of-00004-688fe1305a55e5cc.parquet", "train-00003-of-00004-2d0cd200555ed7fd.parquet"],
            'test' : ['test-00000-of-00001-9c204eb3f4e11791.parquet'],
            'validation' : ["validation-00000-of-00001-cc3c5779fe22e8ca.parquet"]
        }
        
        self.gt_imgs, self.gt_bboxes, self.gt_classes = self.getDataset()
    
    def __len__(self):
        return len(self.gt_imgs)
    
    def __getitem__(self, idx):
        return self.gt_imgs[idx], self.gt_bboxes[idx], self.gt_classes[idx]
    
    def getClassDict(self):
        class_dict = {}
        count = 1
        for class_ in self.classes:
            class_dict[class_] = count
            count += 1
        
        return class_dict
    
    def normaliseToTarget(self, box, width, height):
        return [
            int(box[0] / (width/self.target_width)),
            int(box[1] / (height/self.target_height)),
            int(box[2] / (width/self.target_width)),
            int(box[3] / (height/self.target_height))
        ]
    
    def getSplitPairs(self):
        dataset_list = []
        
        for parq_name in self.splitDict[self.split]:
            dataset_list.append(pd.read_parquet(os.path.join(self.path, parq_name)))
        
        dataset = pd.concat(dataset_list, ignore_index=True)
        
        images = [Image.open(io.BytesIO(img["bytes"])) for img in dataset['image']]
        labels = [label for label in dataset['ground_truth']]
        
        return list(zip(images, labels))
    
    def getTargets(self, labels):
        boxes = []
        classes = []
        vline = labels["valid_line"]
        og_width, og_height = labels['meta']['image_size']['width'], labels['meta']['image_size']['height']
        #Using a nested loop to get all coordinates for given label
        for line in vline: #run through n lines for label
            words = line['words']
            for word in words: #run through n words in each line and retreive the coordinates of said word
                quad = word['quad']
                box = [quad['x1'], quad['y1'], quad['x3'], quad['y3']]
                boxes.append(self.normaliseToTarget(box, og_width, og_height))
                classes.append(0 if not(line['category'] in self.classes) else self.cls_dict[line['category']])

        return boxes, classes
    
    def getDataset(self):
        
        list_of_pairs = self.getSplitPairs()
        
        gt_imgs = []
        gt_bboxes = []
        gt_classes = []
        
        for pair in list_of_pairs:
            img, gt = pair
            img = img.resize((self.target_height, self.target_width))
            
            img_arr = np.asarray(img, dtype='float32') / 255.0
            img_arr = img_arr.transpose(-1,0,1)
            
            gt_imgs.append(img_arr)
            
            boxes, classes = self.getTargets(eval(gt))
                
            gt_bboxes.append(torch.tensor(boxes, dtype=torch.float32))
            gt_classes.append(torch.tensor(classes, dtype=torch.float32))
            
        
        gt_bboxes_padded = pad_sequence(gt_bboxes, batch_first=True, padding_value=0)
        gt_classes_padded = pad_sequence(gt_classes, batch_first=True, padding_value=0)
        
        return gt_imgs, gt_bboxes_padded, gt_classes_padded
