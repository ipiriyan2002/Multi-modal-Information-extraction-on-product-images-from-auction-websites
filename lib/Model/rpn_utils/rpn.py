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
                 score_thresh=0.0,
                 nms_thresh=0.7,
                 pre_nms_k={'TRAIN':12000, 'TEST': 6000},
                 post_nms_k={'TRAIN':2000, 'TEST': 300}
                ):
        
        super(RegionProposalNetwork, self).__init__()
        
        self.rpn_in_channels = rpn_in_channels
        self.conf_score_weight = conf_score_weight
        self.bbox_weight = bbox_weight
        self.num_anchor_boxes = len(anchor_scales) * len(anchor_ratios)
        self.conf_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.pre_nms_k = pre_nms_k
        self.post_nms_k = post_nms_k
        
        # Defining anchor generator and proposal generator
        self.anchorGenerator = AnchorGenerator(anchor_scales=anchor_scales, 
                                               anchor_ratios=anchor_ratios,
                                               stride=stride)
        
        self.proposalGenerator = RPNProposalGenerator(pos_anchor_thresh, neg_anchor_thresh,
                                                      max_samples, anc_ratio)
        
        #Setting up the rpn network
        self.conv1 = nn.Conv2d(self.rpn_in_channels, 512, 3, padding=1)
        self.conf_head = nn.Conv2d(512, self.num_anchor_boxes, 1)
        self.bbox_head = nn.Conv2d(512, self.num_anchor_boxes * 4, 1)
        
        self.init_weights(self.conv1,0, 0.01)
        self.init_weights(self.conf_head,0, 0.01)
        self.init_weights(self.bbox_head,0, 0.01)
        
    def init_weights(self, layer, mean, std):
        layer.weight.data.normal_(mean, std)
        layer.bias.data.zero_()
                    
    def rpnPass(self, featureMaps):
        """
        Given input of feature maps, return confidence scores and rois
        """
        out = self.conv1(featureMaps)
        out = torch.nn.functional.relu(out, inplace=True)
        conf_out = self.conf_head(featureMaps)
        bbox_out = self.bbox_head(featureMaps)
        
        return conf_out, bbox_out
    
    def confScoreLoss(self, batch_size, pred_confs, gt_confs):
        """
        calculate the confidence score loss for only predictions that correspond to positive rois
        """
        gt_confs = gt_confs.type_as(pred_confs).view(-1)
        #weights = torch.abs((gt_confs >= 0).type(torch.int64)).view(-1).type_as(gt_confs)
        pos = torch.nonzero((gt_confs.view(-1) >= 0)).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_confs.view(-1)[pos], gt_confs[pos])
        
        return loss
    
    def bboxLoss(self, batch_size, pred_offsets, gt_offsets, gt_conf_scores):
        """
        Calculate roi bounding box loss for bounding boxes that contain objects and background but not ignore
        """
        gt_offsets = gt_offsets.type_as(pred_offsets).view(-1, 4)
        
        #weights = torch.abs((gt_conf_scores.view(-1) >= 0).type(torch.int64)).view(-1, 1).type_as(gt_offsets)
        pos = torch.nonzero((gt_conf_scores.view(-1) > 0)).squeeze(1)
        total = torch.nonzero((gt_conf_scores.view(-1) >= 0)).squeeze(1)
        
        loss = torch.nn.functional.smooth_l1_loss(pred_offsets.view(-1,4)[pos], gt_offsets[pos], reduction='sum', beta=1/9)
        loss = loss / gt_conf_scores.numel()
        
        return loss
    
    def getProposals(self, anchors, predicted_offsets):
        """
        Given anchors and predicted offsets, generate the proposals
        using rearranged equations from Faster RCNN paper
        https://arxiv.org/pdf/1506.01497.pdf
        """
        anchors = anchors.type_as(predicted_offsets)
        anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')
        
        x_ctr = anchors[...,0] + (predicted_offsets[...,0] * anchors[...,2])
        y_ctr = anchors[...,1] + (predicted_offsets[...,1] * anchors[...,3])
        width = torch.exp(predicted_offsets[...,2]) * anchors[...,2]
        height = torch.exp(predicted_offsets[...,3]) * anchors[...,3]
        
        props = torch.stack([x_ctr, y_ctr, width, height], axis=2)
        
        return ops.box_convert(props, in_fmt='cxcywh', out_fmt='xyxy')
    
    def post_process(self, proposals, scores, img_shapes):
        pre_nms_k = self.pre_nms_k['TRAIN'] if self.training else self.pre_nms_k['TEST']
        post_nms_k = self.post_nms_k['TRAIN'] if self.training else self.post_nms_k['TEST']
        
        #Get the probability given the scores
        scores = torch.sigmoid(scores)
        
        #sort scores and proposals for top scores
        scores, sorted_idxs = torch.sort(scores, dim=-1, descending=True)
        proposals = proposals.view(-1,4)[sorted_idxs.view(-1), :].view(scores.shape[0], -1, 4)
        
        #Get top_k
        proposals = proposals[:, :pre_nms_k, :]
        scores = scores[:, :pre_nms_k]
        
        #Clip boxes to image
        proposals = [ops.clip_boxes_to_image(proposal, img_shape) for proposal, img_shape in zip(proposals, img_shapes)]
        
        #remove small boxes
        keeps = [ops.remove_small_boxes(proposal, 1e-3) for proposal in proposals]
        proposals = [proposal[keep] for proposal, keep in zip(proposals, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        
        #remove boxes with low score
        keeps = [torch.where(score >= self.conf_thresh)[0] for score in scores]
        proposals = [proposal[keep] for proposal, keep in zip(proposals, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        
        #perform nms
        keeps = [ops.nms(proposal,score,self.nms_thresh) for proposal,score in zip(proposals,scores)]
        proposals = [proposal[keep][:post_nms_k] for proposal, keep in zip(proposals, keeps)]
        scores = [score[keep][:post_nms_k] for score, keep in zip(scores, keeps)]
        
        return proposals, scores
    
    def forward(self, feature_maps, images, targets=None):
        """
        Given feature maps, ground truth bounding boxes and classes
        generate rois and roi scores
        return loss dict and rpn out dict
        """
        image_shapes = [img.shape[-2:] for img in images]
                     
        if self.training:
            batch_size = feature_maps.size(dim=0)
            images = [img for img in images]
            gt_bboxes = [target['boxes'] for target in targets]
            gt_classes = [target['labels'] for target in targets]
            

            #generates anchors
            fmSize = feature_maps.shape[-2:]
            anchor = self.anchorGenerator(fmSize)

            gt_conf_scores, gt_offsets = self.proposalGenerator(images, anchor, gt_bboxes, gt_classes)
            
            
            conf_out, bbox_out = self.rpnPass(feature_maps)
            #Predicting proposals and objectness
            
            conf_out = conf_out.permute(0,2,3,1).contiguous().view(batch_size, -1)
            bbox_out = bbox_out.permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
            
            #calculate the loss
            conf_loss = self.confScoreLoss(batch_size, conf_out, gt_conf_scores)
            bbox_loss = self.bboxLoss(batch_size, bbox_out, gt_offsets, gt_conf_scores)
            

            total_loss = {
                "rpn_cls_loss": self.conf_score_weight * conf_loss,
                "rpn_box_loss": self.bbox_weight * bbox_loss
            }

            #Generating proposals
            proposals = self.getProposals(anchor.view(1,-1,4).repeat(batch_size,1,1), bbox_out).view(batch_size, -1, 4)
            
            final_proposals, final_scores = self.post_process(proposals, conf_out, image_shapes)

            rpn_out = {
                "proposals": final_proposals,
                "scores": final_scores
            }

            return total_loss, rpn_out
        else:
            return self.inference(feature_maps, image_shapes)
    
    
    def inference(self, feature_maps, image_shapes):
        """
        Function used do prediction and remove any predictions with score above a threshold and perform nms
        return rpn out dict
        """
        with torch.no_grad():
            batch_size = feature_maps.size(dim=0)
            fmSize = feature_maps.shape[-2:]
            #Generates anchor
            anchor = self.anchorGenerator(fmSize)
            
            #Parse feature maps into rpn
            conf_out, bbox_out = self.rpnPass(feature_maps)
            
            conf_out = conf_out.reshape(batch_size, -1)
            bbox_out = bbox_out.reshape(batch_size, -1, 4)
            
            anchor = anchor.type_as(bbox_out)
            
            #Generate proposals
            proposals = self.getProposals(anchor.view(1,-1,4).repeat(batch_size,1,1), bbox_out)
            
            final_proposals, final_scores = self.post_process(proposals, conf_out, image_shapes)
            
            rpn_out = {
                "proposals": final_proposals,
                "scores": final_scores
            }
            
        return rpn_out


