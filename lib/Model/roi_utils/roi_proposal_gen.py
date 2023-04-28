import torch
import torchvision
import torchvision.ops as ops
import numpy as np
import torch.nn as nn


class ROIProposalGenerator(nn.Module):
    """
    Generate training proposals for the classifier of the Faster RCNN
    """
    def __init__(self, 
                 num_classes, 
                 pos_iou_thresh = 0.7, 
                 neg_iou_thresh = 0.3, 
                 max_samples=256, 
                 ratio_pos_neg=0.25):
        
        super(ROIProposalGenerator, self).__init__()
        
        self.num_classes = num_classes
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.max_samples = max_samples
        self.ratio_pos_neg = ratio_pos_neg
    
    
    def subsample(self, pos_indexes, neg_indexes):
        """
        Given positive indexes and negative indexes subsample such that the number of positive and negative, in total, equals max samples
        Returns the indexes to keep
        """
        
        max_pos_samples = int(self.max_samples * self.ratio_pos_neg)
        
        num_pos = min(pos_indexes.numel(), max_pos_samples)
        
        max_neg_samples = self.max_samples - num_pos
        
        num_neg = min(neg_indexes.numel(), max_neg_samples)
        
        if num_pos > 0 and num_neg > 0:
            
            pos_rand_keep = torch.randperm(pos_indexes.numel(), device=pos_indexes.device)[:num_pos]
        
            neg_rand_keep = torch.randperm(neg_indexes.numel(), device=neg_indexes.device)[:num_neg]
            
            
            pos_samples = pos_indexes[pos_rand_keep]
            neg_samples = neg_indexes[neg_rand_keep]
            
            keep = torch.cat([pos_samples, neg_samples])
        elif num_pos > 0 and num_neg == 0:
            pos_rand_keep = torch.randperm(pos_indexes.numel(), device=pos_indexes.device)[:num_pos]
            
            keep = pos_indexes[pos_rand_keep]
            
        elif num_neg > 0 and num_pos == 0:
            neg_rand_keep = torch.randperm(neg_indexes.numel(), device=neg_indexes.device)[:num_neg]
            
            keep = neg_indexes[neg_rand_keep]
        else:
            raise ValueError("Cannot have no positive and negative boxes")
        
        return keep, num_pos
        
    
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
            proposals = torch.cat((proposals,gt_bboxes), dim=0)

            #Compute the iou matrix
            iou_matrix = ops.box_iou(proposals, gt_bboxes)

            #Max iou ground truth box per proposal
            max_ious, max_ious_indexes = torch.max(iou_matrix, 1)

            #Define storage for labels and ground truth roi for each batch

            #Gather all ground truth boxes into a tensor of shape (number of proposals)
            gt_bbox_pos = max_ious_indexes.reshape(-1, 1,1)
            gt_bbox_pos = gt_bbox_pos.repeat(1,1,4)
            
            target_bboxes = gt_bboxes.view(1, -1,4)
            target_bboxes = target_bboxes.expand(max_ious_indexes.size(-1), -1, 4)
            target_bboxes = torch.gather(target_bboxes, 1, gt_bbox_pos)
            target_bboxes = target_bboxes.view(-1, 4)

            #Gather all classes into a tensor of shape (batch_size, number of proposals)
            gt_classes_pos = max_ious_indexes.reshape(-1, 1)
            target_classes = gt_classes.view(1,-1)
            target_classes = target_classes.expand(max_ious_indexes.size(-1),-1)
            target_classes = torch.gather(target_classes, 1, gt_classes_pos)
            target_classes = target_classes.view(-1)


            #Get the positive indexes as in the foreground boxes
            pos_indexes_positions = max_ious >= self.pos_iou_thresh
            pos_indexes = torch.nonzero(pos_indexes_positions).squeeze(1)
            #Get the negative indexes as in the background boxes (using minimum thresh of 0.1 such that background images are chosen atleast)
            neg_indexes_positions = torch.logical_and(max_ious < self.neg_iou_thresh[0], max_ious >= self.neg_iou_thresh[1])
            neg_indexes = torch.nonzero(neg_indexes_positions).squeeze(1)


            #Subsample and gather the training data
            for batch in range(batch_size):
                keep_index, num_pos_targets = self.subsample(pos_indexes, neg_indexes)

                #Get labels
                gt_labels[index, :keep_index.size(0)] = target_classes[keep_index.view(-1)]
                gt_labels[index, num_pos_targets:] = 0
                #Get proposals
                rois[index, :keep_index.size(0), :] = proposals[keep_index.view(-1), :]
                #Get ground truth boxes
                gt_boxes_per_rois[index, :keep_index.size(0), :] = target_bboxes[keep_index.view(-1), :]

        #Generate the offsets
        rois = ops.box_convert(rois, in_fmt='xyxy', out_fmt='cxcywh')
        gt_boxes_per_rois = ops.box_convert(gt_boxes_per_rois, in_fmt='xyxy', out_fmt='cxcywh')

        tx = (gt_boxes_per_rois[...,0] - rois[...,0]) / rois[...,2]
        ty = (gt_boxes_per_rois[...,1] - rois[...,1]) / rois[...,3]
        tw = torch.log(gt_boxes_per_rois[...,2] / rois[...,2])
        th = torch.log(gt_boxes_per_rois[...,3] / rois[...,3])

        #Gather the offsets
        gt_offsets = torch.stack([tx,ty,tw,th],dim=2).type_as(rois)

        rois = ops.box_convert(rois, in_fmt='cxcywh', out_fmt='xyxy')
        
        return rois, gt_labels, gt_offsets
