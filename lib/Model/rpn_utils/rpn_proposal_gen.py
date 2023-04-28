import torch
import torchvision
import torchvision.ops as ops
import numpy as np
import torch.nn as nn


class RPNProposalGenerator(nn.Module):
    def __init__(self,
                 pos_anc_iou_thresh=0.7, 
                 neg_anc_iou_thresh=0.3, 
                 max_samples=256, 
                 ratio_pos_neg=0.5):
        
        super(RPNProposalGenerator, self).__init__()
        
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
    
    
    def subsample(self, confs):
        """
        Subsample ground truth confidence scores such that the number of positive and negative scores, in total, equals max_samples
        """
        max_pos_samples = int(self.max_samples * self.ratio_pos_neg)
        confs_pos = torch.nonzero(confs.view(-1) ==1).squeeze(1)
        
        num_pos = min(confs_pos.numel(), max_pos_samples)
        
        max_neg_samples = self.max_samples - num_pos
        
        confs_neg = torch.nonzero(confs.view(-1) ==0).squeeze(1)
        
        num_neg = min(confs_neg.numel(), max_neg_samples)
        
        if num_pos > 0 and num_neg > 0:
            
            pos_rand_remove = torch.randperm(confs_pos.numel(), device=confs.device)[num_pos:]
        
            neg_rand_remove = torch.randperm(confs_neg.numel(), device=confs.device)[num_neg:]
            
            
            pos_samples = confs_pos[pos_rand_remove]
            neg_samples = confs_neg[neg_rand_remove]
            
            remove = torch.cat([pos_samples, neg_samples])
            
        elif num_pos > 0 and num_neg == 0:
            pos_rand_remove = torch.randperm(confs_pos.numel(), device=confs.device)[num_pos:]
            
            remove = confs_pos[pos_rand_remove]
            
        elif num_neg > 0 and num_pos == 0:
            neg_rand_remove = torch.randperm(confs_neg.numel(), device=confs.device)[num_neg:]
            
            remove = confs_neg[neg_rand_remove]
        else:
            raise ValueError("Cannot have no positive and negative boxes")

        confs.view(-1)[remove] = -1
        return confs
        
    
    def forward(self, images, in_anchor, all_gt_bboxes, all_gt_orig_classes):
        """
        Given anchor, ground truth bounding boxes and classes, generate training data for the rpn
        """
        batch_size = len(all_gt_bboxes)
        anchors = in_anchor.view(1,-1,4).repeat(batch_size,1,1)
        n_anchors = in_anchor.size(dim=0)
        
        gt_confs_tots = torch.zeros((batch_size, n_anchors)).fill_(-1).type_as(all_gt_orig_classes[0])
        gt_offsets_tots = torch.zeros((batch_size, n_anchors, 4)).type_as(all_gt_bboxes[0])
        
        for index, (anchor, image, gt_bboxes, gt_classes) in enumerate(zip(anchors, images, all_gt_bboxes, all_gt_orig_classes)):
            
            anchor = anchor.type_as(gt_bboxes)
            
            tmSize = image.shape[-2:]
            in_indexes = self.getInImages(anchor, tmSize)
            
            if gt_bboxes.numel() == 0:
                gt_confs_tots[index, in_indexes] = 0
                continue
        
            #Computing a iou matrix of shape (batch_size, number of anchors, number of ground truth boxes)
            #iou_matrix = self.computeIouMatrix(anchor[in_indexes, :], gt_bboxes)
            iou_matrix = ops.box_iou(anchor[in_indexes, :], gt_bboxes)

            #The maximum iou for each anchor with respect to ground truth box
            max_ious, max_ious_indexes = torch.max(iou_matrix, 1)
            #The maximum iou for each ground truth with respect to each anchor inside image
            gt_max_ious, _ = torch.max(iou_matrix, 0)

            #Positive positions for the maximum ious for each ground truth and not 0
            max_iou_pos = torch.logical_and(iou_matrix == gt_max_ious.view(1, -1).expand_as(iou_matrix), iou_matrix >= 0)
            max_iou_pos = torch.sum(max_iou_pos, 1) > 0

            gt_confs = torch.zeros((in_indexes.size(dim=0))).type_as(gt_classes)
            
            #Setting negative subset of anchors
            gt_confs[max_ious < self.neg_anc_iou_thresh] = 0
            
            #Setting discard subset of anchors
            discard_ious_pos = torch.logical_and(max_ious < self.pos_anc_iou_thresh, max_ious >= self.neg_anc_iou_thresh)
            
            gt_confs[discard_ious_pos] = -1

            #Setting positive subset of anchors
            gt_confs[max_iou_pos] = 1
            gt_confs[max_ious >= self.pos_anc_iou_thresh] = 1
            
            #Subsampling to make sure the number of positive samples is <= max samples and vice versa for negative sample
            gt_confs = self.subsample(gt_confs)

            
            #Calculating the ground truth offsets
            #Get anchor-gt_box positions
            max_ious_indexes[max_ious < self.neg_anc_iou_thresh] = 0
            max_ious_indexes[discard_ious_pos] = 0
            
            #gt_offset_pos = max_ious_indexes.reshape(-1,1,1)
            #gt_offset_pos = gt_offset_pos.repeat(1,1,4)
            #Get anchor-gt_classes positions
            #gt_classes_pos = max_ious_indexes.reshape(batch_size,-1,1)

            #target_boxes = gt_bboxes.view(1,-1,4)
            #target_boxes = target_boxes.expand(max_ious_indexes.size(-1), -1,4)
            #Target boxes to calculate offsets
            #target_boxes_gathered = torch.gather(target_boxes, 1, gt_offset_pos)
            target_boxes_gathered = gt_bboxes[max_ious_indexes]
            target_boxes_gathered = ops.box_convert(target_boxes_gathered, in_fmt='xyxy', out_fmt='cxcywh')
            #Anchor boxes to calculate offsets
            anchors_expand = anchor[in_indexes, :].reshape(-1,4).repeat(1,1)
            anchors_expand = ops.box_convert(anchors_expand, in_fmt='xyxy', out_fmt='cxcywh')

            tx = (target_boxes_gathered[...,0] - anchors_expand[...,0]) / anchors_expand[...,2]
            ty = (target_boxes_gathered[...,1] - anchors_expand[...,1]) / anchors_expand[...,3]
            tw = torch.log(target_boxes_gathered[...,2] / anchors_expand[...,2])
            th = torch.log(target_boxes_gathered[...,3] / anchors_expand[...,3])
            #Gather the offsets
            gt_offsets = torch.stack([tx,ty,tw,th],dim=1)

            #Gather the classes
            #target_classes = gt_orig_classes.view(batch_size,1,-1)
            #target_classes = target_classes.expand(batch_size, max_ious_indexes.size(-1),-1)
            #target_classes = torch.gather(target_classes, 2, gt_classes_pos)
            #target_classes = target_classes.view(batch_size, -1)

            #Redefine ground truth confs, offsets and classes to size of rpn out
            #-1 to ignore
            gt_confs_tots[index, in_indexes] = gt_confs

            gt_offsets_tots[index, in_indexes, :] = gt_offsets

            #gt_classes_tots = torch.zeros((batch_size, n_anchors)).type_as(target_classes)
            #gt_classes_tots[:,in_indexes] = target_classes
        
        return gt_confs_tots, gt_offsets_tots
