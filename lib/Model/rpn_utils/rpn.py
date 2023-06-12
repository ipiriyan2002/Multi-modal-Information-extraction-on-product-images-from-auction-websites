import torch
import torchvision
import torchvision.ops as ops
import numpy as np
import torch.nn as nn
from lib.Model.AnchorGen import AnchorGenerator
from lib.Model.rpn_utils.rpn_proposal_gen import RPNProposalGenerator
import math

class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network defined with respect to the Faster RCNN paper
    Generates region of interests and their scores for a given image
    Uses a submodel that parses feature maps and trains from data generated from a proposal generator
    """
    def __init__(self, rpn_in_channels,
                 conf_score_weight = 1, 
                 bbox_weight = 10,
                 pos_anchor_thresh = 0.7, 
                 neg_anchor_thresh = 0.3, 
                 anc_ratio=0.5, 
                 anchor_scales=[8,16,32], 
                 anchor_ratios = [0.5,1,2], 
                 max_samples=256,
                 stride=16,
                 img_max_size=1000,
                 conf_thresh=0.0,
                 nms_thresh=0.7,
                 pre_nms_k={'TRAIN':2000, 'TEST': 1000},
                 post_nms_k={'TRAIN':2000, 'TEST': 1000}
                ):
        
        super(RegionProposalNetwork, self).__init__()
        
        self.rpn_in_channels = rpn_in_channels
        self.conf_score_weight = conf_score_weight
        self.bbox_weight = bbox_weight
        self.num_anchor_boxes = len(anchor_scales) * len(anchor_ratios)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.pre_nms_k = pre_nms_k
        self.post_nms_k = post_nms_k
        
        # Defining anchor generator and proposal generator
        self.anchorGenerator = AnchorGenerator(anchor_scales=anchor_scales, 
                                               anchor_ratios=anchor_ratios,
                                               stride=stride)

        #Using clamp size to limit the input width and height delta into exponential as higher deltas than an expected limit is cauing problems
        self.clamp_size = math.log(img_max_size / stride)
        self.proposalGenerator = RPNProposalGenerator(pos_anchor_thresh, neg_anchor_thresh,
                                                      max_samples, anc_ratio)
        
        #Setting up the rpn network
        self.conv1 = nn.Conv2d(self.rpn_in_channels, 512, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conf_head = nn.Conv2d(512, self.num_anchor_boxes, 1)
        self.bbox_head = nn.Conv2d(512, self.num_anchor_boxes * 4, 1)
        
        self.init_weights(self.conv1,0, 0.01)
        self.init_weights(self.conf_head,0, 0.01)
        self.init_weights(self.bbox_head,0, 0.01)
        
    def init_weights(self, layer, mean, std):
        layer.weight.data.normal_(mean, std)
        layer.bias.data.zero_()

    def parseConv1(self, fmap):

        out = self.conv1(fmap)
        out = self.relu(out)

        return out

    def parseConfs(self, conv_in):

        out = self.conf_head(conv_in)

        _, h, w = out.shape

        c = 1 #Box or not box

        #View in shape (Anchors, Classes, Height, Width)
        #Then permute to shape (Height, Width, Anchors, Classes)
        #Then reshape to shape (Height*Width*Anchors, Classes
        out = out.view(1,-1,c,h,w).permute(0,3,4,1,2).reshape(1, -1, c)
        
        return out

    def parseBoxes(self, conv_in):
        out = self.bbox_head(conv_in)

        ac, h, w = out.shape

        c = 4 #Coordinates

        #View in shape (Batch Num, Anchors, Classes, Height, Width)
        #Then permute to shape (Batch Num, Height, Width, Anchors, Classes)
        #Then reshape to shape (Batch Num, Height*Width*Anchors, Classes
        out = out.view(1,-1,c,h,w).permute(0,3,4,1,2).reshape(1, -1, c)

        return out
        
    def rpnPass(self, featureMaps):
        """
        Given input of feature maps, return confidence scores and rois
        """
        outs = [self.parseConv1(featureMap) for featureMap in featureMaps]
        conf_outs = [self.parseConfs(out) for out in outs]
        bbox_outs = [self.parseBoxes(out) for out in outs]
        
        return torch.cat(conf_outs, dim=1), torch.cat(bbox_outs, dim=1)
    
    def confScoreLoss(self, batch_size, pred_confs, gt_confs):
        """
        calculate the confidence score loss for only predictions that correspond to positive rois
        """
        gt_confs = gt_confs.type_as(pred_confs).view(-1)
        pos = torch.where(gt_confs >= 0)[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_confs.view(-1)[pos], gt_confs[pos])
        
        return loss
    
    def bboxLoss(self, batch_size, pred_offsets, gt_offsets, gt_confs):
        """
        Calculate roi bounding box loss for bounding boxes that contain objects and background but not ignore
        """
        gt_offsets = gt_offsets.type_as(pred_offsets).view(-1, 4)
        
        #weights = torch.abs((gt_conf_scores.view(-1) >= 0).type(torch.int64)).view(-1, 1).type_as(gt_offsets)
        pos = torch.where(gt_confs.view(-1) > 0)[0]
        total = torch.where(gt_confs.view(-1) >= 0)[0]

        loss = torch.nn.functional.smooth_l1_loss(pred_offsets.view(-1,4)[pos], gt_offsets[pos], reduction='sum', beta=1/9)
        loss = loss / total.numel()
        
        return loss
    
    def getProposals(self, anchors, predicted_offsets):
        """
        Given anchors and predicted offsets, generate the proposals
        using rearranged equations from Faster RCNN paper
        https://arxiv.org/pdf/1506.01497.pdf
        """
        anchors = torch.stack(anchors, dim=0)
        anchors = anchors.type_as(predicted_offsets)
        anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')
        
        x_ctr = anchors[...,0] + (predicted_offsets[...,0] * anchors[...,2])
        y_ctr = anchors[...,1] + (predicted_offsets[...,1] * anchors[...,3])
        width = torch.exp(torch.clamp(predicted_offsets[...,2], max=self.clamp_size)) * anchors[...,2]
        height = torch.exp(torch.clamp(predicted_offsets[...,3], max=self.clamp_size)) * anchors[...,3]
        
        props = torch.stack([x_ctr, y_ctr, width, height], axis=2)
        
        return ops.box_convert(props, in_fmt='cxcywh', out_fmt='xyxy')
    
    def post_process(self, proposals, scores, img_shapes):
        #if not(self.training):print("Post RPN Process:")
        pre_nms_k = self.pre_nms_k['TRAIN'] if self.training else self.pre_nms_k['TEST']
        post_nms_k = self.post_nms_k['TRAIN'] if self.training else self.post_nms_k['TEST']
        
        #Get the probability given the scores
        scores = torch.sigmoid(scores)
        
        #sort scores and proposals for top scores
        scores, sorted_idxs = torch.sort(scores, dim=-1, descending=True)
        proposals = proposals.view(-1,4)[sorted_idxs.view(-1), :].view(scores.shape[0], -1, 4)
        #if not(self.training):print(f"Post ordering: {proposals.shape} | {proposals}")
        #Get top_k
        proposals = proposals[:, :pre_nms_k, :]
        scores = scores[:, :pre_nms_k]
        #if not(self.training):print(f"Post PRE NMS K: {proposals.shape} | {proposals}")
        
        #Clip boxes to image
        proposals = [ops.clip_boxes_to_image(proposal, img_shape) for proposal, img_shape in zip(proposals, img_shapes)]

        #if not(self.training):print(f"Post Clipping: {proposals[0].shape} | {proposals[0]}")
        #remove small boxes
        keeps = [ops.remove_small_boxes(proposal, 1e-3) for proposal in proposals]
        proposals = [proposal[keep] for proposal, keep in zip(proposals, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        #if not(self.training):print(f"Post small boxes: {proposals[0].shape} | {proposals[0]}")
        
        #remove boxes with low score
        keeps = [torch.where(score >= self.conf_thresh)[0] for score in scores]
        proposals = [proposal[keep] for proposal, keep in zip(proposals, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        #if not(self.training):print(f"Post low scores: {proposals[0].shape} | {proposals[0]}")
        
        #perform nms
        keeps = [ops.nms(proposal,score,self.nms_thresh) for proposal,score in zip(proposals,scores)]
        proposals = [proposal[keep[:post_nms_k]] for proposal, keep in zip(proposals, keeps)]
        scores = [score[keep[:post_nms_k]] for score, keep in zip(scores, keeps)]

        #if not(self.training):print(f"Post nms: {proposals[0].shape} | {proposals[0]}")
        #if not(self.training):print(f"--------------xxxx--------------")
        
        return proposals, scores
    
    def forward(self, feature_maps, images, targets=None):
        """
        Given feature maps, ground truth bounding boxes and classes
        generate rois and roi scores
        return loss dict and rpn out dict
        """
        image_shapes = [img.shape[-2:] for img in images]
        batch_size = len(images)
        #generates anchors
        fmSizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors = [self.anchorGenerator(fmSize) for fmSize in fmSizes]
                     
        #Predicting proposals and objectness  
        conf_out, bbox_out = self.rpnPass(feature_maps)

        conf_out = conf_out.view(batch_size, -1)
        bbox_out = bbox_out.view(batch_size, -1, 4)
            
        if self.training:
            assert (targets != None), "Expected targets for training"
            
            images = [img for img in images]
            gt_bboxes = [target['boxes'] for target in targets]
            gt_classes = [target['labels'] for target in targets]
            
            gt_conf_scores, gt_offsets = self.proposalGenerator(images, anchors, gt_bboxes, gt_classes)
            
            #calculate the loss
            conf_loss = self.confScoreLoss(batch_size, conf_out, gt_conf_scores)
            bbox_loss = self.bboxLoss(batch_size, bbox_out, gt_offsets, gt_conf_scores)
            

            total_loss = {
                "rpn_cls_loss": self.conf_score_weight * conf_loss,
                "rpn_box_loss": self.bbox_weight * bbox_loss
            }
        else:
            gt_bboxes=None
            gt_classes=None
            gt_offsets=None
            gt_conf_scores=None
            
            total_loss = {"rpn_cls_loss":None, "rpn_box_loss": None}

        #Generating proposals
        proposals = self.getProposals(anchors, bbox_out.detach()).view(batch_size, -1, 4)
        
        final_proposals, final_scores = self.post_process(proposals, conf_out, image_shapes)

        rpn_out = {
            "proposals": final_proposals,
            "scores": final_scores,
            "anchors": anchors
        }

        return total_loss, rpn_out
    


