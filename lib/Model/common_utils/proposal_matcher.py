#Import torch related packages
import torch
import torchvision
import torchvision.ops as ops
import torch.nn as nn

#Import other packages
import numpy

class ProposalMatcher:
    """
    A common proposal matcher that given the proposals/anchors matches to best IoU with gt boxes and returns labels and indexes
    """
    def __init__(self, 
                 pos_thresh=0.7, 
                 neg_thresh=0.3,
                 class_agnostic=True,
                 for_rpn=False
                ):
        
        self.pos_thresh = pos_thresh

        if (isinstance(neg_thresh, list)):
            self.neg_thresh = neg_thresh[0]
            self.min_neg_thresh = neg_thresh[1]
        else:
            self.neg_thresh = neg_thresh
            self.min_neg_thresh = 0.0
            
        self.class_agnostic = class_agnostic
        self.for_rpn = for_rpn
        
    def __call__(self, proposals, boxes, classes):
        """
        Proposals :: Tensor : (number of proposals, 4)
        boxes :: Tensor : (number of boxes, 4)
        classes :: Tensor : (number of boxes,1)
        """

        #Define the common variables
        device = boxes.device
        num_proposals = proposals.shape[0]


        #Define the storage
        labels = torch.zeros((num_proposals), dtype=torch.int64, device=device).fill_(-1)
        proposals = proposals.type_as(boxes)
        
        #Compute the IoU Matrix of shape (number of proposals, number of boxes)
        iou_matrix = ops.box_iou(proposals, boxes)

        #Get the best matched gt box for each proposal
        max_iou_per_proposal, max_iou_per_proposal_indexes = torch.max(iou_matrix, 1)

        #Solve for case with for RPN
        if self.for_rpn:
            #Get the max iou for each gt box wrt proposal
            max_iou_per_gt_box, _ = torch.max(iou_matrix, 0)

            #Get the positions for all low ious
            low_where = torch.where(iou_matrix == max_iou_per_gt_box.view(1, -1))[0]

            #Get the original positions of low quality matches
            low_indexes = max_iou_per_proposal_indexes[low_where]

            #Foreground labels
        foreground_where = max_iou_per_proposal >= self.pos_thresh
        if self.class_agnostic:
            labels[foreground_where] = 1
        else:
            #foreground_where_indexes = foreground_where.type(torch.int64).view(-1, 1)
            #target_classes = classes.view(1,-1).expand(foreground_where.size(0),-1)
            #gathered_classes = torch.gather(target_classes, 0, foreground_where_indexes)
            #labels[foreground_where_indexes.view(-1)] = gathered_classes.view(-1)
            foreground_indexes = max_iou_per_proposal_indexes[foreground_where]
            labels[foreground_where] = classes[foreground_indexes]
            

        #Background labels and indexes
        background_where = torch.logical_and(max_iou_per_proposal < self.neg_thresh, max_iou_per_proposal >= self.min_neg_thresh)
        labels[background_where] = 0
        max_iou_per_proposal_indexes[background_where] = 0

        #Discard labels and indexes
        discard_where = torch.logical_and(max_iou_per_proposal < self.pos_thresh, max_iou_per_proposal >= self.neg_thresh)
        labels[discard_where] = -1
        max_iou_per_proposal_indexes[discard_where] = 0

        #assign low quality foreground labels and indexes if for RPN
        if self.for_rpn:
            if self.class_agnostic:
                labels[low_where] = 1
            else:
                #low_where_indexes = low_where.type(torch.int64).view(-1, 1)
                #target_classes = classes.view(1,-1).expand(low_where_indexes.size(0),-1)
                #gathered_classes = torch.gather(target_classes, 0, low_where_indexes)
                #labels[low_where_indexes.view(-1)] = gathered_classes.view(-1)
                labels[low_where] = classes[low_indexes]
            max_iou_per_proposal_indexes[low_where] = low_indexes

        return labels, max_iou_per_proposal_indexes








        
        
        
        
        
        

