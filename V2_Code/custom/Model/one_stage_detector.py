import torch.nn as nn
import torch
from custom.Model.Backbone_VGG16 import BackboneNetwork
from custom.Model.rpn import RegionProposalNetwork

class OneStageDetector(nn.Module):
    def __init__(self, imgSize, conf_score_weight = 1, bbox_weight = 10,
                 pos_anchor_thresh = 0.7, neg_anchor_thresh = 0.3, anc_ratio=0.5, 
                 anchor_scales=[1,2,3], anchor_ratios = [0.5,1,1.5], stride=1, device=None):
        
        super(OneStageDetector, self).__init__()
        #self.cuda_device = "cuda" if device == None else device
        self.device = device if device != None else torch.device('cpu')
        
        self.img_d, self.img_w, self.img_h = imgSize
        
        #Defining the model
        self.backbone = BackboneNetwork(device=self.device)
        
        empImg = torch.empty(imgSize)
        self.fm_in_channels, self.fm_w, self.fm_h = self.backbone(empImg).shape
        del empImg
        
        self.rpn = RegionProposalNetwork((self.fm_w,self.fm_h), (self.img_w, self.img_h), self.fm_in_channels,
                 conf_score_weight = conf_score_weight, bbox_weight = bbox_weight,
                 pos_anchor_thresh = pos_anchor_thresh, neg_anchor_thresh = neg_anchor_thresh, anc_ratio= anc_ratio, 
                 anchor_scales= anchor_scales, anchor_ratios = anchor_ratios, stride= stride, device=self.device)
    
    def forward(self, images, bboxes, conf_scores):
        out_feature_maps = self.backbone(images)
        
        rpn_loss, rpn_proposals, gt_classes, positive_positions = self.rpn(out_feature_maps, bboxes, conf_scores)
        
        return rpn_loss
    
    def inference(self, images):
        feature_maps = self.backbone(images)
        
        batch_size = images.shape[0]
        
        proposals, conf_scores = self.rpn.inference(feature_maps, batch_size)
        
        return proposals, conf_scores

