import torch
import torchvision
import torchvision.ops as ops
import numpy as np
import torch.nn as nn


class RPNProposalGenerator(nn.Module):
    def __init__(self, fmSize, tmSize, 
                 pos_anc_iou_thresh=0.7, 
                 neg_anc_iou_thresh=0.3, 
                 max_samples=256, 
                 ratio_pos_neg=0.5):
        
        super(RPNProposalGenerator, self).__init__()
        
        #feature map size
        self.fmSize = fmSize
        #Target image size
        self.tmSize = tmSize
        self.pos_anc_iou_thresh = pos_anc_iou_thresh
        self.neg_anc_iou_thresh = neg_anc_iou_thresh
        self.max_samples = max_samples
        self.ratio_pos_neg = ratio_pos_neg
        
    def getInImages(self, anchors, size):
        """
        Given anchors and size limit,
        return indexes to keep such that anchors are within an image
        """
        w, h = size
    
        # Removing anchors box [x1,y1,x3,y3] with any element(x1 | y1 | x3 | y3) with value greater than 0
        minbord = torch.logical_and(anchors[:,[0,2]] >= 0, anchors[:,[1,3]] >= 0)
        # Removing anchors box [x1,y1,x3,y3] with any element(x1 | y1 | x3 | y3) with value lesser than feature map width
        maxbord = torch.logical_and(anchors[:,[0,2]] <= w, anchors[:,[1,3]] <= h)
        
        withinBord = torch.logical_and(minbord, maxbord)
        where_indexes = torch.where(withinBord)[0]
        
        return where_indexes
    
    def generalizeTo(self, bboxes, option='tm2fm'):
        """
        When given bounding boxes and an option,
        return bounding boxes that have been generalized to either feature map size or target image size
        """
        if option == 'tm2fm':
            divide_w, divide_h = self.tmSize
            mult_w, mult_h = self.fmSize
        elif option == 'fm2tm':
            divide_w, divide_h = self.fmSize
            mult_w, mult_h = self.tmSize
        else:
            raise ValueError(f"Expected either tm2fm or fm2tm, got {option} for option parameter")
            
        bboxes_clone = bboxes.clone()
        bboxes_clone[...,[0,2]] /= divide_w
        bboxes_clone[...,[0,2]] *= mult_w
        bboxes_clone[...,[1,3]] /= divide_h
        bboxes_clone[...,[1,3]] *= mult_h
        
        return bboxes_clone
    
    
    def computeIouMatrix(self, anchor, bboxes):
        # given anchor of shape (number of anchors, 4)
        # and bboxes of shape(batch_size, number of bounding boxes, 4)
        # generate a iou matrix of shape (batch_size, number of anchors, number of bounding boxes)
        
        batch_size = bboxes.size(dim=0)
        
        iou_matrix = torch.zeros((batch_size, anchor.size(dim=0), bboxes.size(dim=1))).type_as(bboxes)
        
        for batch in range(batch_size):
            iou_matrix[batch, :, :] = ops.box_iou(anchor, bboxes[batch])
        
        return iou_matrix
    
    def subsample(self, confs):
        """
        Subsample ground truth confidence scores such that the number of positive and negative scores, in total, equals max_samples
        """
        num_pos = torch.sum(confs.view(-1) ==1, dim=0)
        num_neg = torch.sum(confs.view(-1) ==0, dim=0)
        
        max_pos_samples = int(self.max_samples * self.ratio_pos_neg)
        max_neg_samples = self.max_samples - max_pos_samples
        
        if num_pos > max_pos_samples:
            num_pos_idx = torch.where(confs.view(-1) == 1)[0].float()
            replace = num_pos_idx.shape[0] < max_pos_samples
            rand_remove = torch.multinomial(num_pos_idx, num_samples=num_pos_idx.shape[0] - max_pos_samples, replacement=replace)
            confs.view(-1)[rand_remove] = -1
        
        if num_neg > max_neg_samples:
            num_neg_idx = torch.where(confs.view(-1) == 0)[0].float()
            replace = num_neg_idx.shape[0] < max_neg_samples
            rand_remove = torch.multinomial(num_neg_idx, num_samples=num_neg_idx.shape[0] - max_neg_samples, replacement=replace)
            confs.view(-1)[rand_remove] = -1

        return confs
        
    
    def forward(self, anchor, gt_bboxes, gt_orig_classes):
        """
        Given anchor, ground truth bounding boxes and classes, generate training data for the rpn
        """
        anchor = anchor.type_as(gt_bboxes)
        batch_size = gt_bboxes.size(dim=0)
        n_anchors = anchor.size(dim=0)
        
        in_indexes = self.getInImages(anchor, self.tmSize)
        
        #Computing a iou matrix of shape (batch_size, number of anchors, number of ground truth boxes)
        iou_matrix = self.computeIouMatrix(anchor[in_indexes, :], gt_bboxes)
        
        #The maximum iou for each anchor with respect to ground truth box
        max_ious, max_ious_indexes = torch.max(iou_matrix, 2)
        #The maximum iou for each ground truth with respect to each anchor inside image
        gt_max_ious, _ = torch.max(iou_matrix, 1)
        
        #Positive positions for the maximum ious for each ground truth and not 0
        max_iou_pos = torch.logical_and(iou_matrix == gt_max_ious.view(batch_size, 1, -1).expand_as(iou_matrix), iou_matrix > 0)
        max_iou_pos = torch.sum(max_iou_pos, 2) > 0
        
        gt_confs = torch.zeros((batch_size, in_indexes.size(dim=0))).type_as(gt_orig_classes)
        #Setting negative subset of anchors
        gt_confs[max_ious <= self.neg_anc_iou_thresh] = 0
        #Setting positive subset of anchors
        gt_confs[max_iou_pos] = 1
        gt_confs[max_ious >= self.pos_anc_iou_thresh] = 1
        
        #Subsampling to make sure the number of positive samples is <= max samples and vice versa for negative sample
        for batch_num in range(gt_confs.size(dim=0)):
            gt_confs[batch_num] = self.subsample(gt_confs[batch_num])
        
        #Calculating the ground truth offsets
        #Get anchor-gt_box positions
        gt_offset_pos = max_ious_indexes.reshape(batch_size,-1,1,1)
        gt_offset_pos = gt_offset_pos.repeat(1,1,1,4)
        #Get anchor-gt_classes positions
        #gt_classes_pos = max_ious_indexes.reshape(batch_size,-1,1)
        
        target_boxes = gt_bboxes.view(batch_size,1,-1,4)
        target_boxes = target_boxes.expand(batch_size, max_ious_indexes.size(-1), -1,4)
        #Target boxes to calculate offsets
        target_boxes_gathered = torch.gather(target_boxes, 2, gt_offset_pos)
        target_boxes_gathered = ops.box_convert(target_boxes_gathered.view(batch_size,-1,4), in_fmt='xyxy', out_fmt='cxcywh')
        #Anchor boxes to calculate offsets
        anchors_expand = anchor[in_indexes, :].reshape(1,-1,4).repeat(batch_size,1,1)
        anchors_expand = ops.box_convert(anchors_expand, in_fmt='xyxy', out_fmt='cxcywh')
        
        tx = (target_boxes_gathered[...,0] - anchors_expand[...,0]) / anchors_expand[...,2]
        ty = (target_boxes_gathered[...,1] - anchors_expand[...,1]) / anchors_expand[...,3]
        tw = torch.log(target_boxes_gathered[...,2] / anchors_expand[...,2])
        th = torch.log(target_boxes_gathered[...,3] / anchors_expand[...,3])
        #Gather the offsets
        gt_offsets = torch.stack([tx,ty,tw,th],dim=2)
        
        #Gather the classes
        #target_classes = gt_orig_classes.view(batch_size,1,-1)
        #target_classes = target_classes.expand(batch_size, max_ious_indexes.size(-1),-1)
        #target_classes = torch.gather(target_classes, 2, gt_classes_pos)
        #target_classes = target_classes.view(batch_size, -1)
        
        #Redefine ground truth confs, offsets and classes to size of rpn out
        #-1 to ignore
        gt_confs_tots = torch.zeros((batch_size, n_anchors)).fill_(-1).type_as(gt_confs)
        gt_confs_tots[:, in_indexes] = gt_confs
        
        gt_offsets_tots = torch.zeros((batch_size, n_anchors, 4)).type_as(gt_offsets)
        gt_offsets_tots[:, in_indexes, :] = gt_offsets
        
        #gt_classes_tots = torch.zeros((batch_size, n_anchors)).type_as(target_classes)
        #gt_classes_tots[:,in_indexes] = target_classes
        
        return gt_confs_tots, gt_offsets_tots