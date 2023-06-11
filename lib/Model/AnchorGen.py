import torch.nn as nn
import torch
import numpy as np
from torchvision import ops

class AnchorGenerator(nn.Module):
    #Based on https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py
    def __init__(self, anchor_scales=[8,16,32], anchor_ratios=[0.5,1,2], stride=16, base_anchor_size=(16,16)):
        super(AnchorGenerator, self).__init__()

        self.base_anchor_size = base_anchor_size
        self.anchor_scales = torch.tensor(anchor_scales)
        self.anchor_ratios = torch.tensor(anchor_ratios)
        
        self.stride = stride
        
        self.num_sizes = len(anchor_scales) * len(anchor_ratios)
        
    
    #Generate the anchor combinations given a scale and different ratios
    def generateBaseAnchors(self):
        #base_h, base_w = self.base_anchor_size
        #Calculate center point
        #x_ctr = (base_w - 1) * 0.5
        #y_ctr = (base_h - 1) * 0.5
        
        #size = base_h * base_w
        
        #Calculating widths and heights wrt ratios
        #widths = torch.sqrt(size / self.anchor_ratios)
        #heights = widths * self.anchor_ratios

        heights = torch.sqrt(self.anchor_ratios)
        widths = 1 / heights
        #Calculating widths and heights wrt to scales
        widths = widths.expand(self.anchor_scales.size(dim=0),-1) * self.anchor_scales.view(-1,1)
        heights = heights.expand(self.anchor_scales.size(dim=0),-1) * self.anchor_scales.view(-1,1)
        widths = widths.ravel()
        heights = heights.ravel()
        
        #Calculate the xmin, ymin, xmax, ymax
        #xmin = x_ctr - (0.5 * widths)
        #ymin = y_ctr - (0.5 * heights)
        #xmax = x_ctr + (0.5 * widths)
        #ymax = y_ctr + (0.5 * heights)
        
        #out = torch.stack([xmin,ymin,xmax,ymax], axis=1).round()
        out = (torch.stack([-widths, -heights, widths, heights], dim=1) / 2).round()
        return out
    
    def getGrid(self, fmSize):
        fm_h, fm_w = fmSize
        
        #Getting the range of values
        x_range, y_range = torch.arange(0, fm_w), torch.arange(0, fm_h)
        
        #Multiplying by stride to get possible points
        x_range *= self.stride
        y_range *= self.stride
        
        #Getting the meshgrid to compute the possible coordinates
        x_range, y_range = torch.meshgrid(x_range, y_range, indexing='ij')
        
        #Unravel the range into a flattened shape whilst keeping the type
        x_range = x_range.ravel()
        y_range = y_range.ravel()

        out = torch.stack([x_range, y_range, x_range, y_range], dim=1)
        return out
    
    def forward(self, fmSize):
        
        grid = self.getGrid(fmSize).view(-1, 1, 4)
        
        baseAnchors = self.generateBaseAnchors().view(1,-1,4)
        
        anchors = (grid + baseAnchors)
        anchors = anchors.reshape(-1,4)
        
        return anchors