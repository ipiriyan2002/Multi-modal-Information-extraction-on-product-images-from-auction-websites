import torch
import torchvision
import torchvision.ops as ops
import numpy as np
import torch.nn as nn


# Include a nms thershold to reduce number of proposals to train on
# Include a nms thershold to reduce number of proposals to train on
class ProposalGenerator(nn.Module):
    def __init__(self, fmSize, tmSize, 
                 pos_anc_iou_thresh, neg_anc_iou_thresh, 
                 ratio_pos_neg, device=None):
        
        super(ProposalGenerator, self).__init__()
        
        self.fmSize = fmSize
        self.tmSize = tmSize
        self.device = device if device != None else torch.device('cpu')
        
        self.pos_anc_iou_thresh = pos_anc_iou_thresh
        self.neg_anc_iou_thresh = neg_anc_iou_thresh
        #Including a minimum thresh to lower the number of false positive in positive data
        self.min_pos_iou_thresh = 0.5 * self.neg_anc_iou_thresh
        
        self.ratio_pos_neg = ratio_pos_neg
       
    def dropCrossBoundaries(self, anchors):
        fm_w, fm_h = self.fmSize
    
        # Removing anchors box [x1,y1,x3,y3] with any element(x1 | y1 | x3 | y3) with value greater than 0
        minbord = torch.logical_and(anchors[:,0] >= 0, anchors[:,1] >= 0)
        # Removing anchors box [x1,y1,x3,y3] with any element(x1 | y1 | x3 | y3) with value lesser than feature map width
        maxbord = torch.logical_and(anchors[:,2] <= fm_w, anchors[:,3] <= fm_h)
        
        withinBord = torch.logical_and(minbord, maxbord)
        where_indexes = torch.where(withinBord)[0]
        
        return anchors[where_indexes,:], where_indexes
    
    
    def generalizeTo(self, bboxes, option='tm2fm'):
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
    
    #calculates the offsets given the anchor boxes and boxes (ground truth or predicted)
    def getOffsets(self, anchor_boxes, boxes):
        #Change format from (x1,y1,x3,y3) to (cx, cy, w, h)
        anchor_boxes = ops.box_convert(anchor_boxes, in_fmt='xyxy', out_fmt='cxcywh')
        boxes = ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')
        
        #Calculating offsets for center x using formula : (box_x - anchor_x) / anchor_width
        tx = (boxes[...,0] - anchor_boxes[...,0]) / anchor_boxes[...,2]
        #Calculating offsets for center y using formula : (box_y - anchor_y) / anchor_height
        ty = (boxes[...,1] - anchor_boxes[...,1]) / anchor_boxes[...,3]
        #Calculating offsets for width using formula : log(box_w / anchor_width)
        tw = boxes[...,2] / anchor_boxes[...,2]
        tw = torch.log(tw)
        #Calculating offsets for height using formula : log(box_h / anchor_height)
        th = boxes[...,3] / anchor_boxes[...,3]
        th = torch.log(th)
        
        offsets = torch.stack([tx, ty, tw, th], dim=1).to(self.device)
        
        #Deleting local variables to save space
        del tx
        del ty
        del tw
        del th
        
        return offsets
    
    def getMappedBBoxes(self, anchor_boxes, boxes, conf_scores, gt_classes):
        
        anc_bboxes = []
        mapped_classes = []
        
        for index, anchor in enumerate(anchor_boxes):
            anchor = anchor.reshape(-1,4)
            iou_anc_gt = ops.box_iou(anchor, boxes.reshape(-1,4))
            
            iou_pos = iou_anc_gt == conf_scores[index]
            
            bboxes_to_add = boxes[torch.where(iou_pos)[0]]
            classes_to_add = gt_classes[torch.where(iou_pos)[0]]
            
            del iou_anc_gt
            del iou_pos
            
            #only add one gt per anchor
            if bboxes_to_add.numel() != 1: 
                bboxes_to_add = bboxes_to_add[0]
                classes_to_add = classes_to_add[0]
                
            anc_bboxes.append(bboxes_to_add)
            mapped_classes.append(classes_to_add)
            
            del bboxes_to_add
            del classes_to_add
        
        return torch.stack(anc_bboxes), torch.stack(mapped_classes)
              
    def forward(self, anchors, gt_bboxes, gt_orig_classes):
        
        #Assigning the max number of anchors (including positive and negative) 
        #to make sure the number of anchors for each batch is equal
        num_neg_anchors = int(self.ratio_pos_neg * anchors.shape[1])
        num_pos_anchors = anchors.shape[1] - num_neg_anchors
        
        bboxes_generalized = self.generalizeTo(gt_bboxes, option='tm2fm')
        
        all_anchors = []
        gt_confscores = []
        gt_classes = []
        gt_offsets = []
        
        for batch_index, anchor_batch in enumerate(anchors):
            
            anchor_batch, _ = self.dropCrossBoundaries(anchor_batch)
            
            bboxes_batch = bboxes_generalized[batch_index]
            anchor_batch = anchor_batch.type_as(bboxes_batch)
            iou = ops.box_iou(anchor_batch, bboxes_batch)
            flattened_iou = iou.flatten(start_dim=0, end_dim=1)
            
                
            # Using following conditions to get the positions of positive and negative anchors
            # For positive anchors, either get the maximum iou or iou > positive anchor iou threshold
            # For negative anchors, get the non-positie anchors that are below negative anchor iou threshold
            
            pos_anchors_bool_pos = torch.logical_or(torch.logical_and(iou == iou.max(dim=1, keepdim=True)[0], iou > self.min_pos_iou_thresh), 
                                                     iou >= self.pos_anc_iou_thresh)
            
            neg_anchors_bool_pos = torch.logical_and(iou < self.neg_anc_iou_thresh, torch.logical_not(pos_anchors_bool_pos))
            
            
            #Confidence Scores per box
            pos_conf_scores = flattened_iou[torch.where(pos_anchors_bool_pos.flatten(start_dim=0, end_dim=1))[0]].reshape(-1,1)
            
            sorted_pos_conf_scores, sorted_indexes = torch.sort(pos_conf_scores, descending=True, dim=0)
            sorted_indexes = sorted_indexes.to(self.device)
            del pos_conf_scores
            remaining = 0
            
            if sorted_pos_conf_scores.shape[0] < num_pos_anchors:
                remaining = num_pos_anchors - sorted_pos_conf_scores.shape[0]
            else:
                sorted_pos_conf_scores = sorted_pos_conf_scores[:num_pos_anchors]
            
            neg_conf_scores = flattened_iou[torch.where(neg_anchors_bool_pos.flatten(start_dim=0, end_dim=1))[0]].reshape(-1,1)[:num_neg_anchors + remaining]
            
            conf_scores = torch.cat((sorted_pos_conf_scores, neg_conf_scores), dim=0)
            
            del neg_conf_scores
            del sorted_pos_conf_scores
            #all_confscores.append(conf_scores)
            
            del iou
            del flattened_iou
               
            #Anchor Boxes
            
            pos_anchor_boxes = anchor_batch[torch.where(pos_anchors_bool_pos)[0]]
            
            sorted_pos_anchor_boxes = pos_anchor_boxes[sorted_indexes][:num_pos_anchors].reshape(-1,4)

            neg_anchor_boxes = anchor_batch[torch.where(neg_anchors_bool_pos)[0]][:num_neg_anchors + remaining]
            
            anchor_boxes = torch.cat((sorted_pos_anchor_boxes, neg_anchor_boxes), dim=0)
            
            del pos_anchors_bool_pos
            del neg_anchors_bool_pos
            del sorted_pos_anchor_boxes
            
            
            all_anchors.append(anchor_boxes)
            
            #Anchor Classes (with 0 -> not object, 1 -> object)
            pos_objectness = torch.ones_like(torch.empty(pos_anchor_boxes.size(dim=0),1), device=self.device)
            sorted_pos_objectness = pos_objectness[sorted_indexes][:num_pos_anchors].reshape(-1,1)
            
            del pos_objectness
            
            neg_objectness = torch.zeros_like(torch.empty(neg_anchor_boxes.size(dim=0),1), device=self.device)[:num_neg_anchors+remaining]

            del pos_anchor_boxes
            del neg_anchor_boxes
            
            objectness = torch.cat((sorted_pos_objectness, neg_objectness), dim=0)
            
            gt_confscores.append(objectness)
            
            del neg_objectness
            del objectness
            
            #Mapping anchor boxes to ground_truth boxes
            
            mapped_boxes, mapped_classes = self.getMappedBBoxes(anchor_boxes, bboxes_batch, conf_scores, gt_orig_classes[batch_index])
            mapped_boxes = mapped_boxes.to(self.device)
            mapped_classes = mapped_classes.reshape(-1,1).to(self.device)
            
            gt_classes.append(mapped_classes)
            del mapped_classes
            
            gt_offset = self.getOffsets(anchor_boxes, mapped_boxes)
            gt_offsets.append(gt_offset)
            
            del mapped_boxes
            del gt_offset
            
        
        all_anchors = torch.stack(all_anchors).to(self.device)
        gt_confscores = torch.stack(gt_confscores).to(self.device)
        gt_classes = torch.stack(gt_classes).to(self.device)
        gt_offsets = torch.stack(gt_offsets).to(self.device)
        
        return all_anchors, gt_confscores, gt_classes, gt_offsets
