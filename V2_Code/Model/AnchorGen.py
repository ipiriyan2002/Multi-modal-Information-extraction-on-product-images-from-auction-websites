import torch.nn as nn
import torch
import numpy as np
from torchvision import ops

class AnchorGenerator(nn.Module):
    def __init__(self, anchor_scales=[2,4,6], anchor_ratios=[0.5,1,1.5], stride=1, device=None):
        super(AnchorGenerator, self).__init__()
        
        self.device = device if device != None else torch.device('cpu')
        
        self.anchor_scales = anchor_scales
        
        self.anchor_ratios = anchor_ratios
        
        self.stride = stride
        
        self.num_sizes = len(self.anchor_scales) * len(self.anchor_ratios)
        
        self.scale_ratio_comb = zip(self.anchor_scales, [self.anchor_ratios] * len(self.anchor_scales))
        self.baseAnchors = np.array([self.generateBaseAnchor(combo) for combo in self.scale_ratio_comb]).reshape(-1, 4)
        
    
    #Generate the anchor combinations given a scale and different ratios
    def generateBaseAnchor(self, anchorCombo):
        scale, ratios = anchorCombo
        out = []
    
        for i in ratios:
            w = scale * i
            h = scale
        
            base = np.array([-w,-h,w,h]) / 2
            del w
            del h
            out.append(base)
    
        return out
    
    #Given centre coordinates of an anchor point returns the possible bounding boxes for the anchor boxes from initiated scales and ratios
    def applyAnchorsToPoint(self, xc, yc):
        box = [xc,yc] * 2
        anchorBoxesXY = []
        
        for anchor in self.baseAnchors:
            out = np.add(box,anchor)
            #out = np.where(np.add(box, anchor) < 0, 0, np.add(box,anchor))
            anchorBoxesXY.append(out)
            del out
        
        anchorBoxesXY = np.array(anchorBoxesXY)
        return anchorBoxesXY
    
    #Given the feature map size, generate anchor boxes
    def getAnchorBoxes(self, fmSize):
        fm_w, fm_h = fmSize
        
        anchorBoxes = []
        
        for xc in range(fm_w):
            for yc in range(fm_h):
                if (xc + self.stride > fm_w) or (yc + self.stride > fm_h):
                    continue
                box = self.applyAnchorsToPoint(xc+self.stride,yc+self.stride)
                #box = box[np.all(box[...,[0,2]] <= fm_w, axis=1)]
                #box = box[np.all(box[...,[1,3]] <= fm_h, axis=1)]
                #box[...,[0,2]] = np.where(box[...,[0,2]] > fm_w, fm_w, box[...,[0,2]])
                #box[...,[1,3]] = np.where(box[...,[1,3]] > fm_h, fm_h, box[...,[1,3]])
                anchorBoxes.append(box)
                del box
        
        anchorBoxes = np.array(anchorBoxes).reshape(1, -1, 4)
        #anchorBoxes = anchorBoxes[np.all(anchorBoxes >= 0, axis=2)].reshape(1,-1,4)
        #anchorBoxes = anchorBoxes[np.all(anchorBoxes[...,[0,2]] <= fm_w, axis=2)].reshape(1,-1,4)
        #anchorBoxes = anchorBoxes[np.all(anchorBoxes[...,[1,3]] <= fm_h, axis=2)].reshape(1,-1,4)
        
        return anchorBoxes
    
    
    def forward(self, fmSize, batch_size):
        
        anchorBoxes = self.getAnchorBoxes(fmSize)
        anchorBoxes = np.repeat(anchorBoxes, batch_size, axis=0)
        anchorBoxes = torch.tensor(anchorBoxes, device=self.device)
        anchorBoxes = anchorBoxes.reshape(batch_size, -1, 4)
        
        
        return anchorBoxes