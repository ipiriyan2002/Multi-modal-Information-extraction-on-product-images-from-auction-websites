import torch
import torchvision.ops as ops
import torch.nn as nn
#Custom Packages
from lib.Model.common_utils.proposal_matcher import ProposalMatcher
from lib.Model.common_utils.sample_matcher import SampleMatcher


class RPNProposalGenerator(nn.Module):
    def __init__(self,
                 pos_anc_iou_thresh=0.7, 
                 neg_anc_iou_thresh=0.3, 
                 max_samples=256, 
                 ratio_pos_neg=0.5):
        
        super(RPNProposalGenerator, self).__init__()

        self.proposal_matcher = ProposalMatcher(pos_anc_iou_thresh, neg_anc_iou_thresh, class_agnostic=True,for_rpn=True)
        self.sample_matcher = SampleMatcher(max_samples, ratio_pos_neg, keep_inds=False)
        
    def getInImages(self, anchors, size):
        """
        Given anchors and size limit,
        return indexes to keep such that anchors are within an image
        """
        h,w = size
    
        # Removing anchors box [x1,y1,x3,y3] with any element(x1 | y1 | x3 | y3) with value greater than 0
        minbord = torch.logical_and(anchors[:,[0,2]] >= 0, anchors[:,[1,3]] >= 0)
        # Removing anchors box [x1,y1,x3,y3] with any element(x1 | y1 | x3 | y3) with value lesser than feature map width
        maxbord = torch.logical_and(anchors[:,[0,2]] <= w, anchors[:,[1,3]] <= h)
        
        withinBord = torch.logical_and(minbord, maxbord)
        where_indexes = torch.where(withinBord)[0]
        
        return where_indexes
        
    def getMaxAnchorSize(self, anchors):
        anchor_sizes = [anchor.size(0) for anchor in anchors]
        return max(anchor_sizes)

    def forward(self, images, anchors, all_gt_bboxes, all_gt_orig_classes):
        """
        Given anchor, ground truth bounding boxes and classes, generate training data for the rpn
        """
        batch_size = len(images)
        n_anchors = self.getMaxAnchorSize(anchors)
        
        gt_confs_tots = torch.zeros((batch_size, n_anchors)).fill_(-1).type_as(all_gt_orig_classes[0])
        gt_offsets_tots = torch.zeros((batch_size, n_anchors, 4)).type_as(all_gt_bboxes[0])
        
        for index, (anchor, image, gt_bboxes, gt_classes) in enumerate(zip(anchors, images, all_gt_bboxes, all_gt_orig_classes)):

            if gt_bboxes.numel() == 0:
                gt_confs_tots[index] = 0
                continue
            
            anchor = anchor.type_as(gt_bboxes)

            max_anchors = anchor.size(0)

            #tmSize = image.shape[-2:]
            #in_indexes = self.getInImages(anchor, tmSize)

            #Using proposal matcher to generate training proposals for labels and indexes used for bounding boxes
            gt_confs, indexes = self.proposal_matcher(anchor, gt_bboxes, gt_classes)

            #Using sample matcher to sample the indexes to remove from gt_confs
            pos_indexes = torch.nonzero(gt_confs > 0).squeeze(1)
            neg_indexes = torch.nonzero(gt_confs == 0).squeeze(1)
            remove, _, _ = self.sample_matcher(pos_indexes, neg_indexes)

            gt_confs[remove] = -1
            indexes[remove] = 0

            #Gather the GT bounding boxes given the samples indexes
            target_boxes_gathered = gt_bboxes[indexes]
            #Anchor boxes to calculate offsets
            anchors_gathered = anchor.reshape(-1,4).repeat(1,1)

            #Convert format from [xmin, ymin, xmax, ymax] to [x center, y center, width, height]
            target_boxes_gathered = ops.box_convert(target_boxes_gathered, in_fmt='xyxy', out_fmt='cxcywh')
            anchors_gathered = ops.box_convert(anchors_gathered, in_fmt='xyxy', out_fmt='cxcywh')

            #Calculate the offsets
            tx = (target_boxes_gathered[...,0] - anchors_gathered[...,0]) / anchors_gathered[...,2]
            ty = (target_boxes_gathered[...,1] - anchors_gathered[...,1]) / anchors_gathered[...,3]
            tw = torch.log(target_boxes_gathered[...,2] / anchors_gathered[...,2])
            th = torch.log(target_boxes_gathered[...,3] / anchors_gathered[...,3])
            
            #Gather the offsets
            gt_offsets = torch.stack([tx,ty,tw,th],dim=1)

            #Store data in batch tensor
            gt_confs_tots[index, :max_anchors] = gt_confs
            gt_offsets_tots[index, :max_anchors] = gt_offsets
        
        return gt_confs_tots, gt_offsets_tots
