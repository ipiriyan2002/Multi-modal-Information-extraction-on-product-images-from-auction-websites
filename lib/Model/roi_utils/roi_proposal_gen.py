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
                 ratio_pos_neg=0.5,
                 normalize_mean=(0.0,0.0,0.0,0.0),
                 normalize_std =(0.1,0.1,0.2,0.2)):
        
        super(ROIProposalGenerator, self).__init__()
        
        self.num_classes = num_classes
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.max_samples = max_samples
        self.ratio_pos_neg = ratio_pos_neg
        #Define the mean and standard deviation for 
        self.normalize_mean = torch.tensor(normalize_mean, dtype=torch.float32)
        self.normalize_std = torch.tensor(normalize_std, dtype=torch.float32)
    
    def computeIouMatrix(self, proposals, bboxes):
        # given proposals of shape (batch_size, number of proposals, 4)
        # and bboxes of shape(batch_size, number of bounding boxes, 4)
        # generate matrix of ious of shape (batch_size, number of proposals, number of bounding boxes)
        batch_size = bboxes.size(dim=0)
        
        iou_matrix = torch.zeros((batch_size, proposals.size(dim=1), bboxes.size(dim=1))).type_as(bboxes)
        
        for batch in range(batch_size):
            iou_matrix[batch, :, :] = ops.box_iou(proposals[batch], bboxes[batch])
        
        return iou_matrix
    
    def subsample(self, pos_indexes, neg_indexes, pad_indexes):
        """
        Given positive indexes and negative indexes subsample such that the number of positive and negative, in total, equals max samples
        Returns the indexes to keep
        """
        num_pos = pos_indexes.size(dim=0)
        num_neg = neg_indexes.size(dim=0)
        num_pads = pad_indexes.size(dim=0)
        
        max_pos_samples = int(self.max_samples * self.ratio_pos_neg)
        max_neg_samples = self.max_samples - max_pos_samples
        
        pos_samples = pos_indexes[...,1]
        neg_samples = neg_indexes[...,1]
        
        if num_pos > max_pos_samples:
            replace = num_pos < max_pos_samples
            rand_keep = torch.multinomial(torch.arange(0,num_pos,dtype=torch.float32), num_samples=max_pos_samples, replacement=replace)
            pos_samples = pos_indexes[...,1][rand_keep]
        else:
            max_neg_samples += max_pos_samples - num_pos
        
        if num_neg > max_neg_samples:
            replace = num_neg < max_pos_samples
            rand_keep = torch.multinomial(torch.arange(0,num_neg,dtype=torch.float32), num_samples=max_neg_samples, replacement=replace)
            neg_samples = neg_indexes[...,1][rand_keep]
        
        keep = torch.cat([pos_samples, neg_samples])
        num_pos_targets = pos_samples.size(0)
        
        #Get the random pads
        max_pads = self.max_samples - keep.size(0)
        
        if max_pads > 0:
            pads_rand_keep = torch.multinomial(torch.arange(0,num_pads,dtype=torch.float32), num_samples=max_pads, replacement=(num_pads < max_pads))
            pad_samples = pad_indexes[...,1][pads_rand_keep]
        else:
            pad_samples = torch.tensor([])
        
        return keep, pad_samples, num_pos_targets
        
    
    def forward(self, proposals, gt_bboxes, gt_orig_classes):
        """
        Given proposals, ground truth bounding boxes and classes,
        generate training data for the classifier
        """
        batch_size = gt_bboxes.size(dim=0)
        
        #Make sure proposals are in the same device as gt_bboxes
        proposals = proposals.type_as(gt_bboxes)
        
        #Adding the ground truth to the proposals
        proposals = torch.cat((proposals,gt_bboxes), dim=1)
        
        #Compute the iou matrix
        iou_matrix = self.computeIouMatrix(proposals, gt_bboxes)
        
        #Max iou ground truth box per proposal
        max_ious, max_ious_indexes = torch.max(iou_matrix, 2)
        
        #Define storage for labels and ground truth roi for each batch
        
        #Gather all ground truth boxes into a tensor of shape (batch_size, number of proposals)
        gt_bbox_pos = max_ious_indexes.reshape(batch_size, -1, 1,1)
        gt_bbox_pos = gt_bbox_pos.repeat(1,1,1,4)
        target_bboxes = gt_bboxes.view(batch_size, 1, -1,4)
        target_bboxes = target_bboxes.expand(batch_size, max_ious_indexes.size(-1), -1, 4)
        target_bboxes = torch.gather(target_bboxes, 2, gt_bbox_pos)
        target_bboxes = target_bboxes.view(batch_size, -1, 4)
        
        #Gather all classes into a tensor of shape (batch_size, number of proposals)
        gt_classes_pos = max_ious_indexes.reshape(batch_size, -1, 1)
        target_classes = gt_orig_classes.view(batch_size,1,-1)
        target_classes = target_classes.expand(batch_size, max_ious_indexes.size(-1),-1)
        target_classes = torch.gather(target_classes, 2, gt_classes_pos)
        target_classes = target_classes.view(batch_size, -1)
        
        gt_labels = torch.zeros((batch_size, int(self.max_samples))).type_as(target_classes)
        rois = torch.zeros((batch_size, int(self.max_samples), 4)).type_as(proposals)
        gt_boxes_per_rois = torch.zeros((batch_size, int(self.max_samples), 4)).type_as(gt_bboxes)
        
        #Remove nans from being chosen
        notNan = torch.logical_not(torch.isnan(max_ious))
        #Get the positive indexes as in the foreground boxes
        pos_indexes_positions = torch.logical_and(max_ious >= self.pos_iou_thresh, notNan)
        pos_indexes = torch.nonzero(pos_indexes_positions)
        #Get the negative indexes as in the background boxes (using minimum thresh of 0.1 such that background images are chosen atleast)
        neg_indexes_positions = torch.logical_and(max_ious < self.neg_iou_thresh[0], max_ious >= self.neg_iou_thresh[1])
        neg_indexes_positions = torch.logical_and(neg_indexes_positions, notNan)
        neg_indexes = torch.nonzero(neg_indexes_positions)
        
        #Get bounding boxes to pad the training samples to remove cases of nan error
        pad_indexes_positions = torch.logical_and(max_ious < self.neg_iou_thresh[1], notNan)
        pad_indexes = torch.nonzero(pad_indexes_positions)
        
        #Subsample and gather the training data
        for batch in range(batch_size):
            keep_index, pad_index, num_pos_targets = self.subsample(pos_indexes[pos_indexes[...,0] == batch], neg_indexes[neg_indexes[...,0] == batch], pad_indexes[pad_indexes[...,0] == batch])
            
            #Get labels
            gt_labels[batch, :keep_index.size(0)] = target_classes[batch, keep_index]
            gt_labels[batch, num_pos_targets:] = 0
            #Get proposals
            rois[batch, :keep_index.size(0), :] = proposals[batch, keep_index, :]
            #Get ground truth boxes
            gt_boxes_per_rois[batch, :keep_index.size(0), :] = target_bboxes[batch, keep_index, :]
            
            #Pad the boxes with background data to remove cases of nan error
            if pad_index.size(0) > 0:
                rois[batch, keep_index.size(0):, :] = proposals[batch, pad_index, :]
                gt_boxes_per_rois[batch, keep_index.size(0):, :] = target_bboxes[batch, pad_index, :]
        
        #Generate the offsets
        rois = ops.box_convert(rois, in_fmt='xyxy', out_fmt='cxcywh')
        gt_boxes_per_rois = ops.box_convert(gt_boxes_per_rois, in_fmt='xyxy', out_fmt='cxcywh')
        
        tx = (gt_boxes_per_rois[...,0] - rois[...,0]) / rois[...,2]
        ty = (gt_boxes_per_rois[...,1] - rois[...,1]) / rois[...,3]
        tw = torch.log(gt_boxes_per_rois[...,2] / rois[...,2])
        th = torch.log(gt_boxes_per_rois[...,3] / rois[...,3])
        
        #Gather the offsets
        gt_offsets = torch.stack([tx,ty,tw,th],dim=2).type_as(rois)
        
        #Normalize the offsets
        #Substract mean
        self.normalize_mean = self.normalize_mean.type_as(gt_offsets)
        self.normalize_std = self.normalize_std.type_as(gt_offsets)
        
        gt_offsets = gt_offsets - self.normalize_mean.expand_as(gt_offsets)
        #Divide by standard deviation
        gt_offsets = gt_offsets / self.normalize_std.expand_as(gt_offsets)
        
        rois = ops.box_convert(rois, in_fmt='cxcywh', out_fmt='xyxy')
        
        return rois, gt_labels, gt_offsets
