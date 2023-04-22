import torch
import torchvision
import torchvision.ops as ops
import numpy as np
import torch.nn as nn
from lib.Model.AnchorGen import AnchorGenerator
from lib.Model.rpn_utils.rpn_proposal_gen import RPNProposalGenerator

class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network defined with respect to the Faster RCNN paper
    Generates region of interests and their scores for a given image
    Uses a submodel that parses feature maps and trains from data generated from a proposal generator
    """
    def __init__(self, fmSize, imgTargetSize, rpn_in_channels,
                 conf_score_weight = 1, 
                 bbox_weight = 10,
                 pos_anchor_thresh = 0.7, 
                 neg_anchor_thresh = 0.3, 
                 anc_ratio=0.5, 
                 anchor_scales=[8,16,32], 
                 anchor_ratios = [0.5,1,2], 
                 max_samples=256,
                 stride=16):
        
        super(RegionProposalNetwork, self).__init__()
        
        self.fmSize = fmSize
        self.tmSize = imgTargetSize
        self.rpn_in_channels = rpn_in_channels
        self.conf_score_weight = conf_score_weight
        self.bbox_weight = bbox_weight
        self.num_anchor_boxes = len(anchor_scales) * len(anchor_ratios)
        
        # Defining anchor generator and proposal generator
        self.anchorGenerator = AnchorGenerator(anchor_scales=anchor_scales, 
                                               anchor_ratios=anchor_ratios,
                                               stride=stride)
        
        self.proposalGenerator = RPNProposalGenerator(self.fmSize, self.tmSize, 
                                                      pos_anchor_thresh, neg_anchor_thresh,
                                                      max_samples, anc_ratio)
        
        #Setting up the rpn network
        self.conv1 = nn.Conv2d(self.rpn_in_channels, 512, 3, padding=1)
        self.conf_head = nn.Conv2d(512, self.num_anchor_boxes, 1)
        self.bbox_head = nn.Conv2d(512, self.num_anchor_boxes * 4, 1)
        
    
    def rpnPass(self, featureMaps):
        """
        Given input of feature maps, return confidence scores and rois
        """
        out = self.conv1(featureMaps)
        out = torch.nn.functional.relu(out)
        conf_out = self.conf_head(featureMaps)
        bbox_out = self.bbox_head(featureMaps)
        
        return conf_out, bbox_out
    
    def confScoreLoss(self, batch_size, pred_confs, gt_confs):
        """
        calculate the confidence score loss for only predictions that correspond to positive rois
        """
        gt_confs = gt_confs.type_as(pred_confs)
        weights = torch.abs((gt_confs >= 0).type(torch.int64)).view(batch_size, -1).type_as(gt_confs)
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_confs * weights, gt_confs * weights)
        
        return loss
    
    def bboxLoss(self, batch_size, pred_offsets, gt_offsets, gt_conf_scores):
        
        """
        Calculate roi bounding box loss for bounding boxes that contain objects and background but not ignore
        """
        
        gt_offsets = gt_offsets.type_as(pred_offsets)
        
        pos = (gt_conf_scores == 1).type(torch.int64) * 1
        neg = (gt_conf_scores == 0).type(torch.int64) * 1
        
        weights = torch.abs(pos + neg)
        weights = weights.view(batch_size, -1, 1).type_as(gt_offsets)
        
        loss = torch.nn.functional.smooth_l1_loss(pred_offsets * weights, gt_offsets * weights)
        
        return loss
    
    def getProposals(self, anchors, predicted_offsets):
        """
        Given anchors and predicted offsets, generate the proposals
        using rearranged equations from Faster RCNN paper
        https://arxiv.org/pdf/1506.01497.pdf
        """
        anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')
        
        x_ctr = anchors[...,0] + (predicted_offsets[...,0] * anchors[...,2])
        y_ctr = anchors[...,1] + (predicted_offsets[...,1] * anchors[...,3])
        width = torch.exp(predicted_offsets[...,2]) * anchors[...,2]
        height = torch.exp(predicted_offsets[...,3]) * anchors[...,3]
        
        props = torch.stack([x_ctr, y_ctr, width, height], axis=2)
        
        return ops.box_convert(props, in_fmt='cxcywh', out_fmt='xyxy')
    
    def forward(self, feature_maps, gt_bboxes, gt_classes):
        """
        Given feature maps, ground truth bounding boxes and classes
        generate rois and roi scores
        return loss dict and rpn out dict
        """
        batch_size = gt_bboxes.shape[0]
        
        #generates anchors
        anchor = self.anchorGenerator(self.fmSize)
        anchor = anchor.type_as(gt_bboxes)
        
        gt_conf_scores, gt_offsets = self.proposalGenerator(anchor, gt_bboxes, gt_classes)
        
        conf_out, bbox_out = self.rpnPass(feature_maps) 
        #Predicting proposals and objectness
        
        conf_out = conf_out.view(batch_size, -1)
        bbox_out = bbox_out.view(batch_size, -1, 4)
        
        #calculate the loss
        conf_loss = self.confScoreLoss(batch_size, conf_out, gt_conf_scores)
        bbox_loss = self.bboxLoss(batch_size, bbox_out, gt_offsets, gt_conf_scores)
        
        
        total_loss = {
            "rpn_cls_loss": self.conf_score_weight * conf_loss,
            "rpn_bbox_loss": self.bbox_weight * bbox_loss
        }
        
        #Generating proposals
        proposals = self.getProposals(anchor.view(1,-1,4).repeat(batch_size,1,1), bbox_out)
        
        rpn_out = {
            "proposals": proposals,
            "scores": conf_out
        }
        
        return total_loss, rpn_out
    
    
    def inference(self, feature_maps):
        """
        Function used do prediction and remove any predictions with score above a threshold and perform nms
        return rpn out dict
        """
        with torch.no_grad():
            batch_size = feature_maps.size(dim=0)
            #Generates anchor
            anchor = self.anchorGenerator(self.fmSize)
            
            #Parse feature maps into rpn
            conf_out, bbox_out = self.rpnPass(feature_maps)
            
            conf_out = conf_out.reshape(batch_size, -1)
            bbox_out = bbox_out.reshape(batch_size, -1, 4)
            
            anchor = anchor.type_as(bbox_out)
            
            #Generate proposals
            proposals = self.getProposals(anchor.view(1,-1,4).repeat(batch_size,1,1), bbox_out)
            
            rpn_out = {
                "proposals": bbox_out,
                "scores": conf_out
            }
            
        return rpn_out


