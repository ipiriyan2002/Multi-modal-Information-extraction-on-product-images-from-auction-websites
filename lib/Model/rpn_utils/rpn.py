import torch
import torchvision
import torchvision.ops as ops
import numpy as np
import torch.nn as nn
from lib.Model.AnchorGen import AnchorGenerator
from lib.Model.rpn_utils.rpn_proposal_gen import RPNProposalGenerator

class RegionProposalNetwork(nn.Module):
    def __init__(self, fmSize, imgTargetSize, rpn_in_channels,
                 conf_score_weight = 1, bbox_weight = 10,
                 pos_anchor_thresh = 0.7, neg_anchor_thresh = 0.3, anc_ratio=0.5, 
                 anchor_scales=[1,2,3], anchor_ratios = [0.5,1,1.5], min_samples=256,stride=1, device=None):
        super(RegionProposalNetwork, self).__init__()
        #self.cuda_device = "cuda" if device == None else device
        self.device = device if device != None else torch.device('cpu')
        
        self.fmSize = fmSize
        self.tmSize = imgTargetSize
        
        self.rpn_in_channels = rpn_in_channels
        
        self.conf_score_weight = conf_score_weight
        self.bbox_weight = bbox_weight
        
        self.num_anchor_boxes = len(anchor_scales) * len(anchor_ratios)
        
        # Defining anchor generator and proposal generator
        self.anchorGenerator = AnchorGenerator(anchor_scales=anchor_scales, anchor_ratios=anchor_ratios,
                                               stride=stride, device=self.device)
        
        self.proposalGenerator = RPNProposalGenerator(self.fmSize, self.tmSize, 
                                                   pos_anchor_thresh, neg_anchor_thresh,
                                                   min_samples, anc_ratio, self.device)
        
        #Setting up the rpn network
        self.conv1 = nn.Conv2d(rpn_in_channels, 512, 3, padding=1, device=self.device)
        self.conf_head = nn.Conv2d(512, self.num_anchor_boxes, 1, device=self.device)
        self.bbox_head = nn.Conv2d(512, self.num_anchor_boxes * 4, 1, device=self.device)
        
    
    def updateConfWeight(val):
        self.conf_weight = val
    
    def updateBBoxWeight(val):
        self.bbox_weight = val
    
    def rpnPass(self, featureMaps):
        out = self.conv1(featureMaps)
        out = torch.nn.functional.relu(out)
        conf_out = self.conf_head(featureMaps)
        bbox_out = self.bbox_head(featureMaps)
        
        return conf_out, bbox_out
    
    def confScoreLoss(self, batch_size, pred_confs, gt_confs):
        weights = torch.abs((gt_confs == 1).type(torch.int64)).view(batch_size, -1)
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_confs, gt_confs * weights)
        
        return loss
    
    def bboxLoss(self, batch_size, pred_offsets, gt_offsets, gt_conf_scores):
        
        pos = (gt_conf_scores == 1).type(torch.int64) * 1
        neg = (gt_conf_scores == 0).type(torch.int64) * 1
        
        weights = torch.abs(pos + neg)
        weights = weights.view(batch_size, -1, 1)
        pred_offsets = pred_offsets * weights
        gt_offsets = gt_offsets * weights
        loss = torch.nn.functional.smooth_l1_loss(pred_offsets, gt_offsets)
        
        return loss
    
    def getProposals(self, anchors, predicted_offsets):
        anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')
        
        x_ctr = anchors[...,0] + (predicted_offsets[...,0] * anchors[...,2])
        y_ctr = anchors[...,1] + (predicted_offsets[...,1] * anchors[...,3])
        width = torch.exp(predicted_offsets[...,2]) * anchors[...,2]
        height = torch.exp(predicted_offsets[...,3]) * anchors[...,3]
        
        props = torch.stack([x_ctr, y_ctr, width, height], axis=2)
        
        return ops.box_convert(props, in_fmt='cxcywh', out_fmt='xyxy')
    
    def forward(self, feature_maps, gt_bboxes, gt_classes):
        batch_size = gt_bboxes.shape[0]
        
        anchor = self.anchorGenerator(self.fmSize)
        
        gt_conf_scores, gt_offsets = self.proposalGenerator(anchor, gt_bboxes, gt_classes)
        
        conf_out, bbox_out = self.rpnPass(feature_maps) 
        #Predicting proposals and objectness
        
        conf_out = conf_out.view(batch_size, -1)
        bbox_out = bbox_out.view(batch_size, -1, 4)
        
        conf_loss = self.confScoreLoss(batch_size, conf_out, gt_conf_scores)
        bbox_loss = self.bboxLoss(batch_size, bbox_out, gt_offsets, gt_conf_scores)
        
        
        total_loss = {
            "rpn_cls_loss": self.conf_score_weight * conf_loss,
            "rpn_bbox_loss": self.bbox_weight * bbox_loss
        }
        
        #Generating proposals
        proposals = self.getProposals(anchor.view(1,-1,4).repeat(batch_size,1,1), bbox_out)
        #proposals = self.proposalGenerator.generalizeTo(proposals, 'fm2tm')
        
        rpn_out = {
            "proposals": proposals,
            "scores": conf_out
        }
        
        return total_loss, rpn_out
    
    
    def inference(self, feature_maps, nms_thresh=0.7, conf_thresh=0.5):
        with torch.no_grad():
            batch_size = feature_maps.size(dim=0)
            anchor = self.anchorGenerator(self.fmSize)
            
            conf_out, bbox_out = self.rpnPass(feature_maps)
            
            conf_out = conf_out.reshape(batch_size, -1)
            bbox_out = bbox_out.reshape(batch_size, -1, 4)
            
            proposals = self.getProposals(anchor.view(1,-1,4).repeat(batch_size,1,1), bbox_out)
            #proposals = self.proposalGenerator.generalizeTo(proposals, 'fm2tm')
            
            in_confs = [torch.where(conf >= conf_thresh)[0] for conf in conf_out]
            
            confs = [conf[in_conf] for conf,in_conf in zip(conf_out, in_confs)]
            props = [prop[in_conf] for prop,in_conf in zip(proposals, in_confs)]
            
            nms_pos = [ops.nms(prop,conf,nms_thresh) for prop,conf in zip(props,confs)]
            
            final_proposals = [prop[nms] for prop, nms in zip(props,nms_pos)]
            final_confs = [conf[nms] for conf, nms in zip(confs, nms_pos)]
            
            rpn_out = {
                "proposals":final_proposals,
                "scores": final_confs
            }
            
        return final_proposals, final_confs


