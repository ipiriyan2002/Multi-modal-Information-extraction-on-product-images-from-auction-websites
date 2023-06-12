#Import torch packages
import torch
from torchvision import ops
from torch.nn.utils.rnn import pad_sequence
import torch.utils as tu
#Import custom packages
import lib.Loaders.loader_utils as lu
#Import other packages
import os, random, io
import numpy as np
import pandas as pd
from PIL import Image

"""
Classes in Cord dataset:

['menu.cnt', 'menu.discountprice', 'menu.etc', 'menu.itemsubtotal',

 'menu.nm', 'menu.num', 'menu.price', 'menu.sub.cnt', 'menu.sub.nm',
 
 'menu.sub.price', 'menu.sub.unitprice', 'menu.unitprice', 'menu.vatyn',
 
 'sub_total.discount_price', 'sub_total.etc', 'sub_total.othersvc_price', 'sub_total.service_price',
 
 'sub_total.subtotal_price', 'sub_total.tax_price', 'total.cashprice', 'total.changeprice', 
 
 'total.creditcardprice', 'total.emoneyprice', 'total.menuqty_cnt', 'total.menutype_cnt', 
 
 'total.total_etc', 'total.total_price', 'void_menu.nm', 'void_menu.price']
"""
def getClasses(type_):
    """
    Return:
        classes (list) : all available classes in cord dataset
    """
    classes = ['background','menu.cnt', 'menu.discountprice', 'menu.etc', 'menu.itemsubtotal',
               'menu.nm', 'menu.num', 'menu.price', 'menu.sub.cnt', 'menu.sub.nm', 'menu.sub.price',
               'menu.sub.unitprice', 'menu.unitprice', 'menu.vatyn', 'sub_total.discount_price', 'sub_total.etc',
               'sub_total.othersvc_price', 'sub_total.service_price', 'sub_total.subtotal_price', 'sub_total.tax_price',
               'total.cashprice', 'total.changeprice', 'total.creditcardprice', 'total.emoneyprice', 'total.menuqty_cnt',
               'total.menutype_cnt', 'total.total_etc', 'total.total_price', 'void_menu.nm', 'void_menu.price']
        
    text_object_classes = ['background', 'text']
        
    main_classes = ['background', 'menu', 'sub_total', 'total', 'void_menu']
        
    price_classes = ['background', 'others', 'price']
        
    out = {'text':text_object_classes, 'main':main_classes, 'price':price_classes, 'all':classes}
    return out[type_]


class CordDataset(tu.data.Dataset):
    def __init__(self, path, split, class_type="all", pad=False, transform=None):
        """
        Arguments:
            path (string): path to directory
            classes (list): list of all classes available in the cord dataset
            target_size (HeightxWidhtxDepth): size for resizing images
            split (string): split type of dataset
                            Either 'train', 'test' or 'validation'
        """
        assert (split.lower() in ['train', 'test', 'validation']) 
        assert (class_type.lower() in ['text', 'main','price', 'all'])
        
        
        self.path = path
        self.class_type = class_type
        self.classes = getClasses(self.class_type)
        self.num_classes = len(self.classes)
        self.cls_dict = self.getClassDict()
        self.split = split.lower() 
        self.pad = pad
        self.transform = transform
        
        #self.mean = torch.tensor([0.0,0.0,0.0])
        #self.std = torch.tensor([0.0,0.0,0.0])
        #self.pixel_count = 0
        
        #Defining the split dict, with each value contains a list containing file names
        self.splitDict = {
            'train' : ["train-00000-of-00004-b4aaeceff1d90ecb.parquet", "train-00001-of-00004-7dbbe248962764c5.parquet",
                       "train-00002-of-00004-688fe1305a55e5cc.parquet", "train-00003-of-00004-2d0cd200555ed7fd.parquet"],
            'test' : ['test-00000-of-00001-9c204eb3f4e11791.parquet'],
            'validation' : ["validation-00000-of-00001-cc3c5779fe22e8ca.parquet"]
        }
        
        #Get the dataset
        self.gt_imgs, self.gt_targets = self.getDataset()
        
        #self.mean = self.mean / self.pixel_count
        #self.std = torch.sqrt((self.std / self.pixel_count) - (self.mean ** 2))
        
    
    def __len__(self):
        """
        Return:
            size of dataset
        """
        return len(self.gt_imgs)
    
    def __getitem__(self, idx):
        """
        Arguments:
            idx: index of image to retreive
        Return:
            image (torch tensor) (Depth x Height x Width)
            targets (dictionary) (dictionary containing training targets)
        """
        if self.transform != None:
            return self.transform(self.gt_imgs[idx], self.gt_targets[idx])
        
        return self.gt_imgs[idx], self.gt_targets[idx]
    

    def getCategory(self, label):
        """
        Get the category depending on the class_type
        """
        if self.class_type.lower() == 'all':
            return label
        elif self.class_type.lower() == 'main':
            return label.split(".")[-1] if label != 'background' else 'background'
        elif self.class_type.lower() == 'price':
            if label == 'background':
                return label
            
            if 'price' in label:
                return 'price'
            else:
                return 'others'
        else:
            return 'text' if label != 'background' else 'background'
    
    def getClassDict(self):
        """
        Return the numerical value of each class in the dataset as a dictionary
        Return:
            Classes dictionary: (key:value) (classes name: numerical value for class)
        """
        class_dict = {}
        for class_value_, class_ in enumerate(self.classes):
            class_dict[class_] = class_value_
        
        return class_dict
    
    def getSplitPairs(self):
        """
        - Read the split files
        - Read the images and ground truth files
        Return:
            zipped list: a list of tuples (images, labels)
        """
        dataset_list = []
        #Read all parquet files and add the pandas dataframe to a list
        for parq_name in self.splitDict[self.split]:
            dataset_list.append(pd.read_parquet(os.path.join(self.path, parq_name)))
        #Concat all the dataframes into one dataframe with respect to elements and not index
        dataset = pd.concat(dataset_list, ignore_index=True)
        
        #List of all images, read from bytes
        images = [Image.open(io.BytesIO(img["bytes"])) for img in dataset['image']]
        #List of all labels in a dictionary form
        labels = [eval(label) for label in dataset['ground_truth']]
        
        return list(zip(images, labels))
    
    
    def getTargets(self, labels):
        """
        Read and return boxes and classes for the label of an image
        Arguments:
            labels (Dict) : evaluated label read from parquet file
        Return:
            boxes (list) : list of all bounding boxes
            classes (list) : list of all classes
        """
        boxes = []
        classes = []
        texts = []
        vline = labels["valid_line"]
        #Get original width and height of image
        #og_width, og_height = labels['meta']['image_size']['width'], labels['meta']['image_size']['height']
        #Using a nested loop to get all coordinates for given label
        for line in vline: #run through n lines for label
            words = line['words']
            for word in words: #run through n words in each line and retreive the coordinates of said word
                quad = word['quad']
                box = [quad['x1'], quad['y1'], quad['x3'], quad['y3']]
                #norm_box = lu.normaliseToTarget(box, (og_height, og_width), (self.target_height, self.target_width))
                if lu.isBox(box):
                    #Append normalised bounding boxes and numerical value of class
                    boxes.append(box)
                    category = self.getCategory(line['category'])
                    classes.append(self.cls_dict[category])
                    texts.append(word['text'])

        return boxes, classes, texts
        
    def getDataset(self):
        """
        Return:
            gt_imgs (list) : list of tensors of shape (Depth x Height x Width)
            gt_targets (list): list of dictionaries containing bounding boxes, labels, area of boundning boxes, ignore value and image id
        """
        #Get the list of pairs (images, labels)
        list_of_pairs = self.getSplitPairs()
        
        gt_imgs = []
        gt_bboxes = []
        gt_classes = []
        gt_texts = []
        #Run through all pairs
        for pair in list_of_pairs:
            img, gt = pair
            #Image processing
            #img = img.resize((self.target_height, self.target_width)) #Resize image to target height and width
            img_arr = np.asarray(img, dtype='float32') / 255.0 #Normalise array
            img_arr = img_arr.transpose(-1,0,1) #Transpose array to form (Depth x Height x Width)
            img_arr = torch.from_numpy(img_arr)
             #Append tensor form from numpy array
            
            #Add the pixel_counts to calculate mean and std for dataset
            #self.mean += img_arr.sum(axis=[1,2])
            #self.std += (img_arr ** 2).sum(axis=[1,2])
            
            #self.pixel_count += img_arr.shape[1] * img_arr.shape[2]
            
            #Label processing
            boxes, classes, texts = self.getTargets(gt)
            #Append tensor form of boxes and classes with dtype set to float32
            #Only add images with more than one ground truth image
            if len(boxes) > 0:
                gt_imgs.append(img_arr)
                gt_bboxes.append(torch.as_tensor(boxes, dtype=torch.float32))
                gt_classes.append(torch.as_tensor(classes, dtype=torch.int64))
                gt_texts.append(texts)
            
            
        #Pad boxes and classes such that all elements are of same shape
        #-1 -> ignore when training
        if self.pad:
            gt_bboxes = pad_sequence(gt_bboxes, batch_first=True, padding_value=-1)
            gt_classes = pad_sequence(gt_classes, batch_first=True, padding_value=-1)
        
        #Putting all the data into dictionary format
        gt_targets = lu.packTargets(gt_bboxes, gt_classes, gt_texts)
        
        return gt_imgs, gt_targets


