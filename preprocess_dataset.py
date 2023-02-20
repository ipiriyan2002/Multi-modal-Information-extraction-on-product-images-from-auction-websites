#!/usr/bin/env python
# coding: utf-8

# In[79]:


get_ipython().run_line_magic('matplotlib', 'inline')
#inline magic command to make sure all pictures are in line with each cell

#Installing libraries
print("INSTALLING LIBRARIES....")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches
import random
import io
print("LIBRARIES INSTALLED!")


# In[63]:

# Constant Values to use in function
dataset_path = "./cord-v2/data/"
IMG_SIZE = 256
MAX_LABELS = 22




# In[133]:


def setMaxLabels(val):
    MAX_LABELS = val

def setImgSize(size):
    IMG_SIZE = size

"""
Getter function to get the training dataset given the source path
Parameters:
src_path : String

Return:
images : numpy array format for each image in the training dataset
labels : dict form of the labels for each image
"""
def getTrainDataset(src_path):
    train_files = ["train-00000-of-00004-b4aaeceff1d90ecb.parquet", "train-00001-of-00004-7dbbe248962764c5.parquet",
                   "train-00002-of-00004-688fe1305a55e5cc.parquet", "train-00003-of-00004-2d0cd200555ed7fd.parquet"]
    
    
    train1 = pd.read_parquet(src_path + train_files[0])
    train2 = pd.read_parquet(src_path + train_files[1])
    train3 = pd.read_parquet(src_path + train_files[2])
    train4 = pd.read_parquet(src_path + train_files[3])

    train_dataset = pd.concat([train1, train2, train3, train4], ignore_index=True)
    
    byte_dicts = train_dataset["image"]
    labels_pd = train_dataset["ground_truth"]
    images = [Image.open(io.BytesIO(img["bytes"])) for img in byte_dicts]
    labels = [label for label in labels_pd]
    
    return images, labels


# In[136]:


"""
Getter function to get the validation dataset given the source path
Parameters:
src_path : String

Return:
images : numpy array format for each image in the validation dataset
labels : dict form of the labels for each image
"""
def getValDataset(src_path):
    val_file = "validation-00000-of-00001-cc3c5779fe22e8ca.parquet"
    
    val_dataset = pd.read_parquet(src_path + val_file)
    
    byte_dicts = val_dataset["image"]
    labels_pd = val_dataset["ground_truth"]
    images = [Image.open(io.BytesIO(img["bytes"])) for img in byte_dicts]
    labels = [label for label in labels_pd]
    
    return images, labels


# In[137]:


"""
Getter function to get the Test dataset given the source path
Parameters:
src_path : String

Return:
images : numpy array format for each image in the Test dataset
labels : dict form of the labels for each image
"""
def getTestDataset(src_path):
    test_file = "test-00000-of-00001-9c204eb3f4e11791.parquet"
    
    test_dataset = pd.read_parquet(src_path + test_file)
    
    byte_dicts = test_dataset["image"]
    labels_pd = test_dataset["ground_truth"]
    images = [Image.open(io.BytesIO(img["bytes"])) for img in byte_dicts]
    labels = [label for label in labels_pd]
    
    return images, labels


# In[134]:


train_images, train_labels = getTrainDataset(dataset_path)


# A label for training data when evaluated using **eval** function gives the following keys:
# 'gt_parse', 'meta', 'valid_line', 'roi', 'repeating_symbol', 'dontcare'
# 
# We do not need: 'roi', 'repeating_symbol' and 'dontcare
# 
# gt_parse: Tells us information that we do not need for the problem of region detection
# meta: Tells us information such as width, height, id of image
# valid_line: **IMPORTANT** for this problem as it gives us all the coordinates of each text point in image

# In[139]:


"""
Defining function to retreive [x1, y1, x3, y3] coordinates of each datapoint
Retreiving coordinates with category == menu.price
"""
def retreiveCoordinates(labels):
    coords = []
    vline = labels["valid_line"]
    #Using a nested loop to get all coordinates for given label
    for line in vline: #run through n lines for label
        if (line["category"] == "menu.price"): # Retreiving only prices from labels
            words = line['words']
            for word in words: #run through n words in each line and retreive the coordinates of said word
                quad = word['quad']
                box = [quad['x1'], quad['y1'], quad['x3'], quad['y3']]
                coords.append(box)
    
    return coords


# In[140]:


"""
Defining function to get (min x1, min y1, max x3, max y3)
"""

def singleBoxCoord(label):
    coords = retreiveCoordinates(label)
    x1 = [coord[0] for coord in coords] #Get all x1
    y1 = [coord[1] for coord in coords] #Get all y1
    x3 = [coord[2] for coord in coords] #Get all x3
    y3 = [coord[3] for coord in coords] #Get all y3
    
    PAD = 20 #pad the bounding boxes to get dense box
    
    singleBox = [min(x1)-PAD, min(y1)-PAD, max(x3)+PAD, max(y3)+PAD] #define the dense bounding box
    return singleBox


# In[141]:


"""
Defining a displaying function to display images with dense box on them
"""
def showDBox(img_index):
    img = train_images[img_index]
    img_label = eval(train_labels[img_index])
    width, height = img_label['meta']['image_size']['width'], img_label['meta']['image_size']['height']
    dBox_img = singleBoxCoord(img_label)
    fig, ax = plt.subplots() #make a subplot

    ax.imshow(img) #draw the image on the axis of the subplots
    x1 = dBox_img[0]
    y1 = dBox_img[1]
    x3 = dBox_img[2]
    y3 = dBox_img[3]
    rect = patches.Rectangle((x1, y1), (x3-x1), (y3-y1), linewidth=1, edgecolor='g', facecolor='none')
    #Draw a rectangle using bounding boxes 
    ax.add_patch(rect)

    plt.title("WxH: ({0},{1}) | {2}".format(width, height, dBox_img))
    plt.axis('off')
    plt.show()


# In[142]:


"""
Defining a displaying function to display images with dense box on them
"""
def showBBox(img_index):
    img = train_images[img_index]
    img_label = eval(train_labels[img_index])
    width, height = img_label['meta']['image_size']['width'], img_label['meta']['image_size']['height']
    bbox_coords = retreiveCoordinates(img_label)
    fig, ax = plt.subplots() #make a subplot

    ax.imshow(img) #draw the image on the axis of the subplots
    
    for bbox in bbox_coords:
        x1, y1, x3, y3 = bbox
        rect = patches.Rectangle((x1, y1), (x3-x1), (y3-y1), linewidth=1, edgecolor='g', facecolor='none')
        #Draw a rectangle using bounding boxes 
        ax.add_patch(rect)

    plt.title("WxH: ({0},{1})".format(width, height))
    plt.axis('off')
    plt.show()


# In[178]:


"""
Defining pre-process function for images
"""
def pre_process_images(images):
    resized_images = [img_to_array(image.copy().resize((IMG_SIZE, IMG_SIZE))) for image in images]
    
    np_images = np.array(resized_images, dtype='float32') / 255.0
    return np_images


# In[151]:


"""
Defining a function to return the IoU loss of two bboxes
"""
def inter_over_union(dBoxA, dBoxB):
    inter_x1 = max(dBoxA[0], dBoxB[0])
    inter_y1 = max(dBoxA[1], dBoxB[1])
    inter_x3 = min(dBoxA[2], dBoxB[2])
    inter_y3 = min(dBoxA[3], dBoxB[3])
    
    if (inter_x3 < inter_x1) or (inter_y3 < inter_y1):
        return 0
    
    inter_area = (inter_x3 - inter_x1) * (inter_y3 - inter_y1)
    
    union_A = (dBoxA[2] - dBoxA[0]) * (dBoxA[3] - dBoxA[1])
    union_B = (dBoxB[2] - dBoxB[0]) * (dBoxB[3] - dBoxB[1])
    
    union_area = (union_A + union_B) - inter_area
    
    return (inter_area / union_area)


# In[184]:


"""
Generates a bbox defined around the centre coordinate

Done so for the CORD dataset as the prices are not available at the centre at this BBox can be used to get the negative bboxes for the image training data
"""
def generateBBox(width, height):
    centre = [0.5 * width, 0.5 * height]
    
    padd_w = width / 100 if width > 1000 else width / 10
    padd_h = height / 100 if height > 1000 else height / 10
    
    return [centre[0], centre[1], centre[0] + padd_w, centre[1] + padd_h]


# In[185]:


# Returns a list of random numbers given a value
# Used to randomly return a list of coordinates given the width/height of an image as a base
def genListRandNumber(max_val):
    
    list_of_vals = []
    while len(list_of_vals) < 3:
        list_of_vals = [random.randint(0, max_val) for i in range(10)]
        list_of_vals = list(set(list_of_vals))
    
    return list_of_vals

# Returns a randomly generated DBox
# Used to get a negative class (0) DBox for an image
def getNegDBox(width, height):
    xs = genListRandNumber(width)
    ys = genListRandNumber(height)
    
    x1, y1 = min(xs), min(ys)
    x3, y3 = max(xs), max(ys)
    
    return [x1, y1, x3, y3]

# Generates the negative class (0) DBox given the positive class DBox(1) using Intersection-over-Union method
def generateNegBox(dBox, width, height, multiple=False):
    negBox = [0,0,0,0]
    iou = 1
    
    while iou > 0.3:
        negBox = getNegDBox(width, height)
        if not(multiple):
            iou = inter_over_union(dBox, negBox)
        else:
            ious = [inter_over_union(box, negBox) for box in dBox]
            #print(dBox)
            l_iou = len(ious)
            iou = sum(ious) / l_iou
    
    return negBox

"""
Generates N number of negative boxes and also deals with cases of no positive boxes
"""
def generateNNegBox(count, bBoxes, width, height):
    
    negBoxes = []
    randBox = bBoxes
    if (len(bBoxes) == 0):
        randBox = [generateBBox(width, height)]
    
    for i in range(count):
        negBox = generateNegBox(randBox, width, height, multiple=True)
        
        while negBox in negBoxes:
            negBox = generateNegBox(randBox, width, height, multiple=True)
        
        negBoxes.append(negBox)
    
    return negBoxes


# In[186]:


def normalize(bbox, width, height):
    return [
        bbox[0] / width,
        bbox[1] / height,
        bbox[2] / width,
        bbox[3] / height
           ]

def unnormalize(bbox, width, height):
    return [
        bbox[0] * width,
        bbox[1] * height,
        bbox[2] * width,
        bbox[3] * height
    ]

# defining a function to up-sample the 1 confidence bboxes with handling for cases with 0 1 confidence boxes
# Returns the bboxes and confidence scores in list format
def duplicateOneConf(bboxes, width, height):
    if (len(bboxes) >= MAX_LABELS):
        outBoxes = bboxes[:MAX_LABELS]
        outScores = [1] * MAX_LABELS
        return outBoxes, outScores
    elif (len(bboxes) == 0):
        negBoxes = generateNNegBox(MAX_LABELS, [], width, height)
        outBoxes = negBoxes
        outScores = [0] * MAX_LABELS
        return outBoxes, outScores
        
    dupBoxes = bboxes * (int(MAX_LABELS / len(bboxes)) + 1)
    
    outBoxes = dupBoxes[:MAX_LABELS]
    outScores = [1] * MAX_LABELS
    
    return outBoxes, outScores
    


# In[190]:


"""
Defining pre-process function for labels
"""

"""
Preprocess function for a single big box
"""
def pre_process_label_single(label):
    pos_and_negs = []
    conf_scores = []
    ev_lab = eval(label)
    width, height = ev_lab['meta']['image_size']['width'], ev_lab['meta']['image_size']['height']
    dBox = singleBoxCoord(ev_lab)
    #normalize the dbox coordinates using original width and height such that the values are between 0 and 1
    pos_labels = normalize(dBox, width, height)
    
    pos_and_negs.append(pos_labels) #Adding the positive (correct dBox coordinates) sample | 1 denotes correct dBox
    conf_scores.append(1)
    negBox = generateNegBox(dBox, width, height)
    
    neg_labels = normalize(negBox, width, height)
    

    pos_and_negs.append(neg_labels)
    conf_scores.append(0)
        
    return pos_and_negs, conf_scores

"""
Preprocess function for multiple bboxes
"""
def pre_process_label_dense(label):
    ev_lab = eval(label)
    width, height = ev_lab['meta']['image_size']['width'], ev_lab['meta']['image_size']['height']
    retBoxes = retreiveCoordinates(ev_lab)
    num_pos_boxes = len(retBoxes)
    conf_scores = [1] * num_pos_boxes
    bBoxes_norm = [normalize(box, width, height) for box in retBoxes]
    
    outBoxes = []
    outScores = []
    remaining = MAX_LABELS - num_pos_boxes
    

    if (num_pos_boxes >= MAX_LABELS):
        return bBoxes_norm[:MAX_LABELS], conf_scores[:MAX_LABELS]
    else:
        # Deal with cases where bBoxes_norm has len = 0
        outBoxes.extend(bBoxes_norm)
        outScores.extend(conf_scores)
        remaining = MAX_LABELS - len(bBoxes_norm)
    
    nNegBoxes = generateNNegBox(remaining, retBoxes, width, height)
    nNegBoxes_norm = [normalize(box, width, height) for box in nNegBoxes]
    nConf_scores = [0] * len(nNegBoxes_norm)
    outBoxes.extend(nNegBoxes_norm)
    outScores.extend(nConf_scores)
    
    return outBoxes, outScores

"""
Common preprocessing function that does either dense or single box label preprocessing
"""
def pre_process_labels(labels, dense=False):
    new_labels = []
    confScores = []
    for label in labels:
        #pre-process each label
        if not(dense):
            nLabel, confScore = pre_process_label(label)
            new_labels.extend(nLabel)
            confScores.extend(confScore)
        else:
            nLabel, confScore = pre_process_label_dense(label)
            new_labels.append(nLabel)
            confScores.append(confScore)
        
    new_labels = np.array(new_labels)
    confScores = np.array(confScores)
    
    return new_labels, confScores

