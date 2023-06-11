import torch
import torchvision.ops as ops
import torch.nn as nn

#Custom Packages
from lib.Model.common_utils.proposal_matcher import ProposalMatcher
from lib.Model.common_utils.sample_matcher import SampleMatcher

class ROIProposalGenerator(nn.Module):
    """
    Generate training proposals for the classifier of the Faster RCNN
    """
    def __init__(self, 
                 num_classes, 
                 pos_iou_thresh = 0.7, 
                 neg_iou_thresh = 0.3, 
                 max_samples=256, 
                 ratio_pos_neg=0.25,
                 weights=(10.0,10.0,5.0,5.0)
                ):
        
        super(ROIProposalGenerator, self).__init__()
        
        self.num_classes = num_classes
        self.weights = weights
        self.max_samples = max_samples
        self.proposal_matcher = ProposalMatcher(pos_iou_thresh, neg_iou_thresh, class_agnostic=False,for_rpn=False)
        self.sample_matcher = SampleMatcher(max_samples, ratio_pos_neg, keep_inds=True)


    def forward(self, all_proposals, all_gt_bboxes, all_gt_orig_classes):
        """
        Given proposals, ground truth bounding boxes and classes,
        generate training data for the classifier
        """
        batch_size = len(all_proposals)
        
        gt_labels = torch.zeros((batch_size, int(self.max_samples))).type_as(all_gt_orig_classes[0])
        rois = torch.zeros((batch_size, int(self.max_samples), 4)).type_as(all_proposals[0])
        gt_boxes_per_rois = torch.zeros((batch_size, int(self.max_samples), 4)).type_as(all_gt_bboxes[0])
        
        for index, (proposals, gt_bboxes, gt_classes) in enumerate(zip(all_proposals, all_gt_bboxes, all_gt_orig_classes)):
            
            if gt_bboxes.numel() == 0:
                continue
            
            #Make sure proposals are in the same device as gt_bboxes
            proposals = proposals.type_as(gt_bboxes)

            #Adding the ground truth to the proposals
            proposals = torch.cat((proposals,gt_bboxes))

            #Using proposal matcher to generate training proposals for labels and indexes used for bounding boxes
            target_classes, indexes = self.proposal_matcher(proposals, gt_bboxes, gt_classes)

            #Gather all ground truth boxes into a tensor of shape (number of proposals)
            target_bboxes = gt_bboxes[indexes]

            #Get the positive indexes as in the foreground boxes
            pos_indexes = torch.nonzero(target_classes > 0).squeeze(1)
            #Get the negative indexes as in the background boxes
            neg_indexes = torch.nonzero(target_classes == 0).squeeze(1)

            #Subsample and gather the training data
            keep, num_pos, _ = self.sample_matcher(pos_indexes, neg_indexes)

            #Get labels
            gt_labels[index, :keep.size(0)] = target_classes[keep]
            gt_labels[index, num_pos:] = 0
            #Get proposals
            rois[index, :keep.size(0), :] = proposals[keep, :]
            #Get ground truth boxes
            gt_boxes_per_rois[index, :keep.size(0), :] = target_bboxes[keep, :]

        #Generate the offsets
        rois = ops.box_convert(rois, in_fmt='xyxy', out_fmt='cxcywh')
        gt_boxes_per_rois = ops.box_convert(gt_boxes_per_rois, in_fmt='xyxy', out_fmt='cxcywh')

        tx = self.weights[0] * (gt_boxes_per_rois[...,0] - rois[...,0]) / rois[...,2]
        ty = self.weights[1] * (gt_boxes_per_rois[...,1] - rois[...,1]) / rois[...,3]
        tw = self.weights[2] * torch.log(gt_boxes_per_rois[...,2] / rois[...,2])
        th = self.weights[3] * torch.log(gt_boxes_per_rois[...,3] / rois[...,3])

        #Gather the offsets
        gt_offsets = torch.stack([tx,ty,tw,th],dim=2).type_as(rois)

        rois = ops.box_convert(rois, in_fmt='cxcywh', out_fmt='xyxy')
        
        return rois, gt_labels, gt_offsets
