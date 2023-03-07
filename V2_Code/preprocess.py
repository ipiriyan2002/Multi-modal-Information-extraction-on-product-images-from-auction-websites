#from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import random
import utils
import math

# Return the factors for a given number
# Used to get the width and height pair given an area
# num : Int
# Returns : List of Tuples (Width, Height)
def getWidthHeight(num):
    pairs = []
    for i in range(1, num + 1):
        if num % i == 0:
            j = num // i
            if ((i,j) in pairs or (j,i) in pairs):
                continue
            pairs.append((i,j))
    
    return pairs

# Returns the Area of a given bounding box
def getBoxArea(box):
    return (box[3] - box[1]) * (box[2] - box[0])

# Return a correct pair of width and height by making sure the width and height of the box is within img size
# If not, then use a defualt pair of width and height (Only for cases where the condition is not met)
# pairs : List of Tuples (Width, Height)
# width, height : Int, Int
# Returns : Tuple (Width, Height)
def getCorrectPair(pairs, width, height):
    pairs = pairs[::-1]
    count = 0
    try:
        pair = pairs[count]
    except:
        return (50,50)
    while (pair[0] > width) or (pair[1] > height) or (pair[0] < 20 and pair[1] > 40) or (pair[1] < 20 and pair[0] > 40):
        count += 1
        try:
            pair = pairs[count]
        except:
            pair = (50,50)
    
    return pair
    
    
# Generate a bounding box which is within a defined range withing 50% of the average area
# widht, height : Int, Int
# bboxes : list of bounding boxes -> [[x1,y1,x3,y3],[x1,y1,x3,y3]]
# Returns : bounding box
def generateBox(width, height, bboxes):
    areas = np.array([getBoxArea(box) for box in bboxes])

    avg_area = np.mean(areas)

    avg_area = 2500 if math.isnan(avg_area) else avg_area
    
    min_area, max_area = int(avg_area * 0.5), int(avg_area * 1.5)
    
    random_area = random.randint(min_area, max_area)
    
    area_factor = getWidthHeight(random_area)

    w, h = getCorrectPair(area_factor, width, height)
    x1 = random.randint(1,(width - w))
    y1 = random.randint(1,(height - h))
        
    x3, y3 = x1 + w, y1 + h
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x3 = min(width, x3)
    y3 = min(height, y3)
    
    return [x1,y1,x3,y3]
    
"""
Defining function to retreive [x1, y1, x3, y3] coordinates of each datapoint
Retreiving coordinates with category == menu.price
"""
def retreiveCoordinates(labels):
    coords = {}
    vline = labels['valid_line']
    #Using a nested loop to get all coordinates for given label
    for line in vline: #run through n lines for label
        category = line['category']
        if (not(category in coords.keys())):
            coords[category] = []
        
        words = line['words']
        for word in words: #run through n words in each line and retreive the coordinates of said word
            quad = word['quad']
            box = [quad['x1'], quad['y1'], quad['x3'], quad['y3']]
            coords[category].append(box)
    
    return coords

# Return the size of the image from the meta information
def getSize(label):
    return label['meta']['image_size']['width'], label['meta']['image_size']['height']

# Returns a number (count) of negative boxes that meet the iou threshold ( < iou_thresh)
def generateNegBoxes(width, height, pos_boxes, count, iou_thresh=0.3):
    negBoxes = []
    
    while len(negBoxes) != count:
        negBox = generateBox(width, height, pos_boxes)
        
        ious = utils.listIOU(negBox, pos_boxes)
        ious.append(0)
        
        if max(ious) > iou_thresh:
            continue
        negBoxes.append(negBox)
    
    return negBoxes


# Preprocess the CORD dataset to return a pair of bounding boxes numpy array and confidence scores numpy array
def preprocess_cord_prices(labels, max_labels, input_width, input_height):
    prices_bboxes = []
    conf_scores = []
    
    for label in labels:
        elabel = eval(label)
        # Width and Height of the image
        width, height = getSize(elabel)
        # Retreiving the coordinates in the form of a dictionary with category : bbox pairs
        rc = retreiveCoordinates(eval(label))
        
        # Get the positive boxes, in this case menu.price bounding boxes
        try:
            boxes = rc['menu.price']
            scores = [1] * len(boxes)
            remaining = max_labels - len(boxes)
        except:
            boxes = []
            scores = []
            remaining = max_labels

        neg_boxes = []
        neg_scores = []
        
        # If more boxes are needed, use the custom generator
        if (remaining > 0):
            input_boxes = boxes if len(boxes) > 0 else neg_boxes
            neg_box_n = generateNegBoxes(width, height, input_boxes, remaining) 
            neg_score_n = [0] * remaining
            neg_boxes.extend(neg_box_n)
            neg_scores.extend(neg_score_n)

            remaining = 0
        
        # Add the negative boxes and then normalize the boxes
        boxes.extend(neg_boxes)
        scores.extend(neg_scores)
        boxes_norm = [utils.normalize(box, width, height) for box in boxes]
        #boxes_unnorm = [utils.unnormalize(normBox, input_width, input_height) for normBox in boxes_norm]
        prices_bboxes.append(boxes_norm)
        conf_scores.append(scores)
    
    prices_bboxes = np.array(prices_bboxes)
    conf_scores = np.array(conf_scores)
    return prices_bboxes, conf_scores

def preprocess_images(images, width, height):
    resized_images = [np.array(image.copy().resize((width, height))) for image in images]
    
    np_images = np.array(resized_images, dtype='float32') / 255.0
    return np_images