import torch
import torchvision
import torchvision.ops as ops
import numpy as np
import torch.nn as nn
from Model.AnchorGen import AnchorGenerator

#TODO: Need to change
def calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size):
    target_pos = torch.ones_like(conf_scores_pos)
    target_neg = torch.zeros_like(conf_scores_neg)
    
    target = torch.cat((target_pos, target_neg))
    inputs = torch.cat((conf_scores_pos, conf_scores_neg))
     
    loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, target, reduction='sum') * 1. / batch_size
    
    return loss

#TODO: Need to change
def calc_bbox_reg_loss(gt_offsets, reg_offsets_pos, batch_size):
    assert gt_offsets.size() == reg_offsets_pos.size()
    loss = torch.nn.functional.smooth_l1_loss(reg_offsets_pos, gt_offsets, reduction='sum') * 1. / batch_size
    return loss

class RegionProposalNetwork(nn.Module):
    def __init__(self, backbone, img_height, img_width, subsample_ratio, rpn_in_channels, anchor_scales=[1,2,3], anchor_ratios = [0.5,1,1.5], device=None):
        super(RegionProposalNetwork, self).__init__()
        #self.cuda_device = "cuda" if device == None else device
        self.device = device#torch.device(self.cuda_device if torch.cuda.is_available() else "cpu")
        
        # Feature extractor (We are using self-defined class that sets up VGG16 model as backbone)
        self.backbone = backbone
        self.anchorGenerator = AnchorGenerator(device=self.device)
        self.anchorGenerator = self.anchorGenerator.to(self.device)
        
        # Image input size
        self.img_height = img_height
        self.img_width = img_width
        self.subsampleRatio = subsample_ratio
        # Since using all the layers of VGG16 except last max pooling, the subsample ratio would be set to 16
        self.bbout_height = img_height // self.subsampleRatio
        self.bbout_width = img_width // self.subsampleRatio
        self.rpn_in_channels = rpn_in_channels
        
        
        #Default Anchor Scales and Ratios
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.num_anchor_boxes = len(self.anchor_scales) * len(self.anchor_ratios)
        #Assigning weights
        self.conf_weight = 1
        self.bbox_weight = 5
        
        
        #Setting up the rpn network
        self.conv1 = nn.Conv2d(rpn_in_channels, 512, 3, padding=1)
        self.conv1 = self.conv1.to(self.device)
        self.conf_head = nn.Conv2d(512, self.num_anchor_boxes, 1)
        self.conf_head = self.conf_head.to(self.device)
        self.bbox_head = nn.Conv2d(512, self.num_anchor_boxes * 4, 1)
        self.bbox_head = self.bbox_head.to(self.device)
    
    def updateConfWeight(val):
        self.conf_weight = val
    
    def updateBBoxWeight(val):
        self.bbox_weight = val
        
    def forward(self, imgs, gt_bboxes, gt_conf_scores):
        batch_size = imgs.size(dim=0)
        
        positive_anc_ind, negative_anc_ind, \
        GT_conf_scores, GT_offsets, GT_class_pos, \
        positive_anc_coords, negative_anc_coords, positive_anc_ind_sep = self.anchorGenerator((self.bbout_width, self.bbout_height), 
                                                                                         gt_bboxes, gt_conf_scores)
        
        #Predicting proposals and objectness
        feature_maps = self.backbone(imgs)
        
        x = self.conv1(feature_maps)
        x = torch.nn.functional.relu(x)
        
        conf_out = self.conf_head(x)
        bbox_out = self.bbox_head(x)
        
        conf_score_pos = conf_out.flatten()[positive_anc_ind]
        conf_score_neg = conf_out.flatten()[negative_anc_ind]
        
        offsets_pos = bbox_out.contiguous().view(-1,4)[positive_anc_ind]
        
        #Generating proposals
        anchors = ops.box_convert(positive_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
        anchors = anchors.to(self.device)
        
        proposals_ = torch.zeros_like(anchors)
        proposals_[:,0] = anchors[:,0] + offsets_pos[:,0]*anchors[:,2]
        proposals_[:,1] = anchors[:,1] + offsets_pos[:,1]*anchors[:,3]
        proposals_[:,2] = anchors[:,2] * torch.exp(offsets_pos[:,2])
        proposals_[:,3] = anchors[:,3] * torch.exp(offsets_pos[:,3])

        # change format of proposals back from 'cxcywh' to 'xyxy'
        proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')
        
        cls_loss = calc_cls_loss(conf_score_pos, conf_score_neg, batch_size)
        reg_loss = calc_bbox_reg_loss(GT_offsets, offsets_pos, batch_size)
        
        total_rpn_loss = self.conf_weight * cls_loss + self.bbox_weight * reg_loss
        
        return total_rpn_loss, feature_maps, proposals, positive_anc_ind_sep, GT_class_pos
    
    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        with torch.no_grad():
            batch_size = images.size(dim=0)
            feature_map = self.backbone(images)

            anchors = self.anchorGenerator.getAnchorBoxes((self.bbout_width, self.bbout_height))
            anchors = np.repeat(anchors, batch_size, axis=0)
            anchors = anchors.reshape(batch_size, -1, 4)
            anchors = torch.as_tensor(anchors).to(self.device)
        
            # get conf scores and offsets
            x = self.conv1(feature_map)
            x = torch.nn.functional.relu(x)
            conf_scores_pred = self.conf_head(x)
            offsets_pred = self.bbox_head(x)
            
            conf_scores_pred = conf_scores_pred.reshape(batch_size, -1)
            offsets_pred = offsets_pred.reshape(batch_size, -1, 4)

            # filter out proposals based on conf threshold and nms threshold for each image
            proposals_final = []
            conf_scores_final = []
            for i in range(batch_size):
                conf_scores = torch.sigmoid(conf_scores_pred[i])
                offsets = offsets_pred[i]
                anc_boxes = anchors[i]
                
                #Generating proposals
                anchors_ = ops.box_convert(anc_boxes, in_fmt='xyxy', out_fmt='cxcywh')
                
                proposals_ = torch.zeros_like(anchors_)
                proposals_[:,0] = anchors_[:,0] + offsets[:,0]*anchors_[:,2]
                proposals_[:,1] = anchors_[:,1] + offsets[:,1]*anchors_[:,3]
                proposals_[:,2] = anchors_[:,2] * torch.exp(offsets[:,2])
                proposals_[:,3] = anchors_[:,3] * torch.exp(offsets[:,3])

                # change format of proposals back from 'cxcywh' to 'xyxy'
                proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')
                
                # filter based on confidence threshold
                conf_idx = torch.where(conf_scores >= conf_thresh)[0]
                conf_scores_pos = conf_scores[conf_idx]
                proposals_pos = proposals[conf_idx]
                # filter based on nms threshold
                conf_scores_pos = conf_scores_pos.to(torch.float64)
                nms_idx = ops.nms(proposals_pos, conf_scores_pos, nms_thresh)
                conf_scores_pos = conf_scores_pos[nms_idx]
                proposals_pos = proposals_pos[nms_idx]

                proposals_final.append(proposals_pos)
                conf_scores_final.append(conf_scores_pos)
            
        return proposals_final, conf_scores_final, feature_map
