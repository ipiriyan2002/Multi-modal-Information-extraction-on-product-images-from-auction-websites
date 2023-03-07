import torch.nn as nn
import torch
from Model.Backbone_VGG16 import BackboneNetwork
from Model.rpn import RegionProposalNetwork


class OneStageDetector(nn.Module):
    def __init__(self, img_w, img_h, in_channels, sratio, device=None):
        super(OneStageDetector, self).__init__()
        #self.cuda_device = "cuda" if device == None else device
        self.device = device#torch.device(self.cuda_device if torch.cuda.is_available() else "cpu")
        
        self.img_w = img_w
        self.img_h = img_h
        self.in_channels = in_channels
        self.sratio = sratio
        
        
        self.backbone = BackboneNetwork(device=self.device)
        self.backbone = self.backbone.to(self.device)
        self.rpn = RegionProposalNetwork(self.backbone, self.img_h, self.img_w, self.sratio, self.in_channels, device=self.device)
        self.rpn = self.rpn.to(self.device)
        
        
    
    def forward(self, images, bboxes, conf_scores):
        total_rpn_loss, feature_maps, proposals, positive_anc_ind_sep, GT_class_pos = self.rpn(images, bboxes, conf_scores)
        #Moving tensors to gpu/cpu
        total_rpn_loss = total_rpn_loss.to(self.device)
        feature_maps = feature_maps.to(self.device)
        proposals = proposals.to(self.device)
        positive_anc_ind_sep = positive_anc_ind_sep.to(self.device)
        GT_class_pos = GT_class_pos.to(self.device)
        
        return total_rpn_loss
    
    def inference(self,images,conf_thresh=0.5, nms_thresh=0.7):
        proposals_final, conf_scores_final, feature_map = self.rpn.inference(images, conf_thresh, nms_thresh)
        
        #proposals_final = proposals_final.to(self.device)
        #conf_scores_final = conf_scores_final.to(self.device)
        #feature_map = feature_map.to(self.device)
        
        
        #final_props = []
        #for i in proposals_final:
        #    i = i.to(self.device)
        #    i[...,[0,2]] /= self.sratio
        #    i[...,[0,2]] *= self.img_w
        #    i[...,[1,3]] /= self.sratio
        #    i[...,[1,3]] *= self.img_h
        #    
        #    final_props.append(i)
        
        return proposals_final, conf_scores_final
