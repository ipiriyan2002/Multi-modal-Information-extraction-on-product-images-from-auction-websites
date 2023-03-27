import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


#View image given image data, ground truth (gt) bboxes, gt classes (in numerical form), class dict (key (int) : value (class))
# index of image to print, anchor_boxes included (bool), anchor boxes, and iou threshold 
def displayImg(img_data, bboxes_data, classes_data, clsdict, idx, inc_anc, anc_boxes=[], iou_thresh=0.5):
    print_img = img_data[idx].permute(1,2,0).numpy()
    fig, ax = plt.subplots()

    ax.imshow(print_img)
    plotted_boxes = []
    for index, box in enumerate(bboxes_data[idx]):
        x1, y1, x3, y3 = box
        
        rect = patches.Rectangle((x1,y1), x3-x1, y3-y1, edgecolor='g',facecolor='none')
        ax.add_patch(rect)
        
        if classes_data[idx][index] != 0:
            plotted_boxes.append(box)
            cls_name = clsdict[int(classes_data[idx][index])]
            
            ax.text(x1+10,y1+10, cls_name, bbox=dict(facecolor='green', alpha=1))
    
    if inc_anc:
        for ancbox in anc_boxes:
            for i in plotted_boxes:
                if inter_over_union(i, ancbox) > iou_thresh:
                    x1, y1, x3, y3 = ancbox
            
                    rect = patches.Rectangle((x1,y1), x3-x1, y3-y1, edgecolor='r',facecolor='none')
                    ax.add_patch(rect)
    
    plt.show()