import torch
import torchvision
import torchvision.ops as ops
import numpy as np
import torch.nn as nn
from Model.AnchorGen import AnchorGenerator
from Model.ProposalGen import ProposalGenerator

class RegionProposalNetwork(nn.Module):
    def __init__(self, fmSize, imgTargetSize, rpn_in_channels,
                 conf_score_weight = 1, bbox_weight = 10,
                 pos_anchor_thresh = 0.7, neg_anchor_thresh = 0.3, anc_ratio=0.5, 
                 anchor_scales=[1,2,3], anchor_ratios = [0.5,1,1.5], stride=1, device=None):
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
        
        self.proposalGenerator = ProposalGenerator(self.fmSize, self.tmSize, 
                                                   pos_anchor_thresh, neg_anchor_thresh,
                                                   anc_ratio, self.device)
        
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
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_confs, gt_confs, reduction='sum')
        loss /= batch_size
        
        return loss
    
    def bboxLoss(self, batch_size, pred_offsets, gt_offsets, gt_conf_scores):
        loss = 0
        
        for batch_index, pred_batch in enumerate(pred_offsets):
            gt_offsets_batch = gt_offsets[batch_index]
            gt_cs_batch = gt_conf_scores[batch_index]
            pos_positions = torch.where(gt_cs_batch > 0)
            loss += float(torch.nn.functional.smooth_l1_loss(pred_batch[pos_positions], gt_offsets_batch[pos_positions], reduction='sum'))
            
        loss /= batch_size
        
        return loss
    
    def getProposals(self, anchors, predicted_offsets):
        anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')
        
        #An empty placeholder to put the proposals
        proposals = torch.zeros_like(anchors)
        #Getting the proposals for the cx
        proposals[...,0] = (predicted_offsets[...,0] * anchors[...,2]) + anchors[...,0]
        #Getting the proposals for the cy
        proposals[...,1] = (predicted_offsets[...,1] * anchors[...,3]) + anchors[...,1]
        #Getting the proposals for the w
        proposals[...,2] = torch.exp(predicted_offsets[...,2]) * anchors[...,2]
        #Getting the proposals for the h
        proposals[...,3] = torch.exp(predicted_offsets[...,3]) * anchors[...,3]
        
        return ops.box_convert(proposals, in_fmt='cxcywh', out_fmt='xyxy')
    
    def forward(self, feature_maps, gt_bboxes, gt_classes):
        batch_size = gt_bboxes.shape[0]
        
        anchors = self.anchorGenerator(self.fmSize, batch_size)
        
        all_anchors, gt_conf_scores, gt_classes, gt_offsets = self.proposalGenerator(anchors, gt_bboxes, gt_classes)
        del anchors
        
        conf_out, bbox_out = self.rpnPass(feature_maps) 
        #Predicting proposals and objectness
        
        conf_out = conf_out.reshape(batch_size, -1, 1)
        bbox_out = bbox_out.reshape(batch_size, -1, 4)
        
        conf_loss = self.confScoreLoss(batch_size, conf_out, gt_conf_scores)
        bbox_loss = self.bboxLoss(batch_size, bbox_out, gt_offsets, gt_conf_scores)
        
        del gt_offsets
        del conf_out
        
        total_loss = (self.conf_score_weight * conf_loss) + (self.bbox_weight * bbox_loss)
        
        del conf_loss
        del bbox_loss
        
        #Generating proposals
        proposals = self.getProposals(all_anchors, bbox_out)
        proposals = self.proposalGenerator.generalizeTo(proposals, 'fm2tm')
        
        del all_anchors
        del bbox_out
        
        return total_loss, proposals, gt_classes, gt_conf_scores
    
    def inference(self, feature_maps, batch_size):
        with torch.no_grad():
            anchors = self.anchorGenerator(self.fmSize, batch_size)
            
            conf_out, bbox_out = self.rpnPass(feature_maps)
            
            conf_out = conf_out.reshape(batch_size, -1, 1)
            bbox_out = bbox_out.reshape(batch_size, -1, 4)
            
            proposals = self.getProposals(anchors, bbox_out)
            proposals = self.proposalGenerator.generalizeTo(proposals, 'fm2tm')
            
            del anchors
            del bbox_out
            
        return proposals, conf_out