# Importing packages
import yaml
import os
#from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
from torchvision.datasets import VOCDetection


# Path to config folder
CONFIG_FOLDER_PATH = "./Configs"
DATA_FOLDER_PATH = "./Data"

# Load the config file given the name of the file
def load_config_file(fname):
    if CONFIG_FOLDER_PATH in fname:
        file_path = fname
    else:
        file_path = os.path.join(CONFIG_FOLDER_PATH, fname)
        
    with open(file_path) as f:
        read_config = yaml.safe_load(f)
    
    return read_config

def load_voc_2007(target_dir="./Data/", split="train"):
    assert (split in ["train", "validation", "test"])
    
    save_path = target_dir
    if split == "train":
        save_path = save_path + "VOC2007_TRAIN/"
    elif split == "validation":
        save_path = save_path + "VOC2007_VAL/"
    elif split == "test":
        save_path = save_path + "VOC2007_TEST/"
    
    if not(os.path.exists(target_dir)):
        try:
            os.makedirs(save_path)
        except:
            print("TARGET PATH EXISTS")
    
    _ = VOCDetection(save_path, year='2007', image_set=split, download=True)
    
    remove_paths = ["VOCdevkit/VOC2007/ImageSets/","VOCdevkit/VOC2007/SegmentationObject/","VOCdevkit/VOC2007/SegmentationClass/"]
    
    for rp in remove_paths:
        shutil.rmtree(save_path + rp, ignore_errors=True)
        
# Return a tuple containing the images and ground_truth of the cord dataset
def load_cord(split):
    dataset = load_dataset("naver-clova-ix/cord-v2", split=split)
    
    return dataset["image"], dataset["ground_truth"]


# Plot the image including the boxes given in.
# boxes can be empty to not draw any boxes
def plotImage(image, boxes, width, height, color='g'):
    fig, ax = plt.subplots(figsize=(20, 20))
    
    ax.imshow(image)
    
    for box in boxes:
        x1, y1, x3, y3 = box
        rect = patches.Rectangle((x1, y1), (x3-x1), (y3-y1), linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    
    plt.title("WxH: ({0},{1})".format(width, height))
    plt.axis('off')
    plt.show()

# Plot the image including the boxes given in.
# boxes can be empty to not draw any boxes
def plotImageWithConf(image, boxes, conf,width, height, color1='g',color2='r'):
    fig, ax = plt.subplots(figsize=(20, 20))
    
    ax.imshow(image)
    
    for index, box in enumerate(boxes):
        x1, y1, x3, y3 = box
        color = color1 if conf[index] == 1 else color2
        rect = patches.Rectangle((x1, y1), (x3-x1), (y3-y1), linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    
    plt.title("WxH: ({0},{1})".format(width, height))
    plt.axis('off')
    plt.show()



"""
Defining a function to return the IoU loss of two bboxes
"""
def inter_over_union(boxA, boxB):
    inter_x1 = max(boxA[0], boxB[0])
    inter_y1 = max(boxA[1], boxB[1])
    inter_x3 = min(boxA[2], boxB[2])
    inter_y3 = min(boxA[3], boxB[3])
    
    if (inter_x3 < inter_x1) or (inter_y3 < inter_y1):
        return 0
    
    inter_area = (inter_x3 - inter_x1) * (inter_y3 - inter_y1)
    
    union_A = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    union_B = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    union_area = (union_A + union_B) - inter_area
    
    return (inter_area / union_area)


def listIOU(box, boxList):
    vals = []
    
    for box2 in boxList:
        if (len(box2) != 4):
            continue
        
        val = inter_over_union(box, box2)
        
        vals.append(val)
    
    return vals


# Normalize a bounding box given width and height
def normalize(bbox, width, height):
    return [
        bbox[0] / width,
        bbox[1] / height,
        bbox[2] / width,
        bbox[3] / height
           ]

# Unnormalize a bounding box given width and height
def unnormalize(bbox, width, height):
    return [
        int(bbox[0] * width),
        int(bbox[1] * height),
        int(bbox[2] * width),
        int(bbox[3] * height)
    ]

# Return a tensorflow dataset for training given images, labels, buffer size and batch size
#def getTensorDataset(images, labels, buffer_size, batch):
#    dataset = tf.data.Dataset.from_tensor_slices((images,labels))
#    dataset = dataset.shuffle(buffer_size=buffer_size)
#    dataset = dataset.batch(batch)
#    dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#    
#    return dataset
    
    
