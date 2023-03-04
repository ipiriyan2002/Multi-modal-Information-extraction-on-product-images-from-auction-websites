import albumentations as alb
import numpy as np
import preprocess as pre
import utils


def getTransformedDataset(images, bboxes, cscores, img_width, img_height, max_labels, dupes = 2):
    newImages = []
    newBoxes = []
    newScores = []
    for index, image in enumerate(images):
        for _ in range(dupes):
            tImage, tBoxes, tScores = transformImage(image, bboxes[index], cscores[index], img_width, img_height)
            
            if len(tBoxes) < max_labels:
                remaining = max_labels - len(tBoxes)
                remainingBoxes = pre.generateNegBoxes(img_width, img_height, tBoxes, remaining)
                remainingScores = [0] * remaining
                remainingBoxes_norm = [utils.normalize(box, img_width, img_height) for box in remainingBoxes]
                tBoxes.extend(remainingBoxes_norm)
                tScores.extend(remainingScores)
            
            newImages.append(tImage)
            newBoxes.append(tBoxes)
            newScores.append(tScores)
        
        newImages.append(image)
        newBoxes.append(bboxes[index])
        newScores.append(cscores[index])
    
    newImages = np.array(newImages)
    newBoxes = np.array(newBoxes)
    newScores = np.array(newScores)
    
    return newImages, newBoxes, newScores
        



def transformImage(image, boxes, confs, finalWidth, finalHeight, min_vis=0.45):
    transform = alb.Compose([
        alb.RandomCrop(width=256, height=256),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.RandomRotate90(p=0.3),
        alb.Resize(width=finalWidth, height=finalHeight, p=1)], 
        bbox_params=alb.BboxParams(format='albumentations', min_visibility=0.45, label_fields=['conf_scores']))
    
    tImage = transform(image=image, bboxes=boxes, conf_scores=confs)
    
    return tImage['image'], tImage['bboxes'], tImage['conf_scores']