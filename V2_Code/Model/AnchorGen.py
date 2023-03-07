import torch.nn as nn
import torch
import numpy as np
from torchvision import ops

class AnchorGenerator(nn.Module):
    def __init__(self, anchor_scales=[2,4,6], anchor_ratios=[0.5,1,1.5], device=None):
        super(AnchorGenerator, self).__init__()
        self.device = device
        
        self.anchor_scales = anchor_scales
        
        self.anchor_ratios = anchor_ratios
        
        self.num_sizes = len(self.anchor_scales) * len(self.anchor_ratios)
        
        self.scale_ratio_comb = zip(self.anchor_scales, [self.anchor_ratios] * len(self.anchor_scales))
        self.baseAnchors = np.array([self.generateBaseAnchor(combo) for combo in self.scale_ratio_comb]).reshape(-1, 4)
    
    def generateBaseAnchor(self, anchorCombo):
        scale, ratios = anchorCombo
        out = []
    
        for i in ratios:
            w = scale * i
            h = scale
        
            base = np.array([-w,-h,w,h]) / 2
            out.append(base)
    
        return out
    
    #Given centre coordinates of an anchor point returns the possible bounding boxes for the anchor boxes from initiated scales and ratios
    def applyAnchorsToPoint(self, xc, yc):
        box = [xc,yc] * 2
        anchorBoxesXY = []
        
        for anchor in self.baseAnchors:
            out = np.where(np.add(box, anchor) < 0, 0, np.add(box,anchor))
            anchorBoxesXY.append(out)
            
        return np.array(anchorBoxesXY)
    
    def getAnchorBoxes(self, fmSize):
        fm_w, fm_h = fmSize
        
        anchorBoxes = []
        
        for xc in range(fm_w):
            for yc in range(fm_h):
                box = self.applyAnchorsToPoint(xc+0.5,yc+0.5)
                box[...,[0,2]] = np.where(box[...,[0,2]] > fm_w, fm_w, box[...,[0,2]])
                box[...,[1,3]] = np.where(box[...,[1,3]] > fm_h, fm_h, box[...,[1,3]])
                anchorBoxes.append(box)
        
        return np.array(anchorBoxes).reshape(1, fm_w, fm_w, self.num_sizes, 4)
    
    def get_iou_mat(self, batch_size, anc_boxes_all, gt_bboxes_all):
    
        # flatten anchor boxes
        anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
        anc_boxes_flat = anc_boxes_flat.to(self.device)
        # get total anchor boxes for a single image
        
        tot_anc_boxes = anc_boxes_flat.size(dim=1)
        # create a placeholder to compute IoUs amongst the boxes
        
        ious_mat = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))
        ious_mat = ious_mat.to(self.device)
        
        # compute IoU of the anc boxes with the gt boxes for all the images
        for i in range(batch_size):
            gt_bboxes = gt_bboxes_all[i]
            gt_bboxes = gt_bboxes.to(self.device)
            anc_boxes = anc_boxes_flat[i]
            anc_boxes = anc_boxes.to(self.device)
            ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)

        return ious_mat
    
    def calc_gt_offsets(self, pos_anc_coords, gt_bbox_mapping):
        pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
        pos_anc_coords = pos_anc_coords.to(self.device)
        gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')
        gt_bbox_mapping = gt_bbox_mapping.to(self.device)

        gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
        anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]

        tx_ = (gt_cx - anc_cx)/anc_w
        ty_ = (gt_cy - anc_cy)/anc_h
        tw_ = torch.log(gt_w / anc_w)
        th_ = torch.log(gt_h / anc_h)

        return torch.stack([tx_, ty_, tw_, th_], dim=-1)
    
    def getTrainAnchors(self, anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2):
        anc_boxes_all = anc_boxes_all.to(self.device)
        batch_size, fm_w, fm_h, _, _ = anc_boxes_all.shape
        maxGTBoxes = gt_bboxes_all.shape[1] # max number of groundtruth bboxes in a batch

        # get total number of anchor boxes in a single image
        tot_anc_boxes = fm_w * fm_h * self.num_sizes

        # get the iou matrix which contains iou of every anchor box
        # against all the groundtruth bboxes in an image
        iou_mat = self.get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all)
        iou_mat = iou_mat.to(self.device)
        
        # for every groundtruth bbox in an image, find the iou 
        # with the anchor box which it overlaps the most
        max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True)
        max_iou_per_gt_box = max_iou_per_gt_box.to(self.device)
        # get positive anchor boxes

        # condition 1: the anchor box with the max iou for every gt bbox
        positive_anc_mask = torch.logical_and(iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0)
        
        # condition 2: anchor boxes with iou above a threshold with any of the gt bboxes
        positive_anc_mask = torch.logical_or(positive_anc_mask, iou_mat > pos_thresh)

        positive_anc_ind_sep = torch.where(positive_anc_mask)[0] # get separate indices in the batch
        # combine all the batches and get the idxs of the +ve anchor boxes
        positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)
        positive_anc_ind = torch.where(positive_anc_mask)[0]

        # for every anchor box, get the iou and the idx of the
        # gt bbox it overlaps with the most
        max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
        max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)

        # get iou scores of the +ve anchor boxes
        GT_conf_scores = max_iou_per_anc[positive_anc_ind]

        # get gt classes of the +ve anchor boxes

        # expand gt classes to map against every anchor box
        gt_classes_expand = gt_classes_all.view(batch_size, 1, maxGTBoxes).expand(batch_size, tot_anc_boxes, maxGTBoxes)
        # for every anchor box, consider only the class of the gt bbox it overlaps with the most
        GT_class = torch.gather(gt_classes_expand, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1)
        # combine all the batches and get the mapped classes of the +ve anchor boxes
        GT_class = GT_class.flatten(start_dim=0, end_dim=1)
        GT_class_pos = GT_class[positive_anc_ind]

        # get gt bbox coordinates of the +ve anchor boxes

        # expand all the gt bboxes to map against every anchor box
        gt_bboxes_expand = gt_bboxes_all.view(batch_size, 1, maxGTBoxes, 4).expand(batch_size, tot_anc_boxes, maxGTBoxes, 4)
        # for every anchor box, consider only the coordinates of the gt bbox it overlaps with the most
        GT_bboxes = torch.gather(gt_bboxes_expand, -2, max_iou_per_anc_ind.reshape(batch_size, tot_anc_boxes, 1, 1).repeat(1, 1, 1, 4))
        # combine all the batches and get the mapped gt bbox coordinates of the +ve anchor boxes
        GT_bboxes = GT_bboxes.flatten(start_dim=0, end_dim=2)
        GT_bboxes_pos = GT_bboxes[positive_anc_ind]

        # get coordinates of +ve anc boxes
        anc_boxes_flat = anc_boxes_all.flatten(start_dim=0, end_dim=-2) # flatten all the anchor boxes
        anc_boxes_flat = anc_boxes_flat.to(self.device)
        positive_anc_coords = anc_boxes_flat[positive_anc_ind]

        # calculate gt offsets
        GT_offsets = self.calc_gt_offsets(positive_anc_coords, GT_bboxes_pos)
        GT_offsets = GT_offsets.to(self.device)
        # get -ve anchors

        # condition: select the anchor boxes with max iou less than the threshold
        negative_anc_mask = (max_iou_per_anc < neg_thresh)
        negative_anc_ind = torch.where(negative_anc_mask)[0]
        # sample -ve samples to match the +ve samples
        negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (positive_anc_ind.shape[0],))]
        negative_anc_coords = anc_boxes_flat[negative_anc_ind]

        return positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, positive_anc_coords, negative_anc_coords, positive_anc_ind_sep
    
    def forward(self, fmSize, gt_bboxes, gt_conf_scores):
        fm_w, fm_h = fmSize
        batch_size = gt_bboxes.size(dim=0)
        
        anchorBoxes = self.getAnchorBoxes((fm_w, fm_h))
        anchorBoxes = np.repeat(anchorBoxes, batch_size, axis=0)
        #print(anchorBoxes.shape)
        
        gt_bboxes_clone = gt_bboxes.clone()
        gt_bboxes_clone[...,[0,2]] *= fm_w
        gt_bboxes_clone[...,[1,3]] *= fm_h
        gt_cs_clone = gt_conf_scores.clone()
        
        #positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, 
        #positive_anc_coords, negative_anc_coords, positive_anc_ind_sep = self.getTrainAnchors(gt_bboxes_clone, gt_cs_clone, anchorBoxes)
        
        return self.getTrainAnchors(torch.as_tensor(anchorBoxes).to(self.device), gt_bboxes_clone, gt_cs_clone)
