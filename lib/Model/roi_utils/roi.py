import torch
import torchvision
from lib.Model.roi_utils.roi_proposal_gen import ROIProposalGenerator

class ROINetwork(torch.nn.Module):
    """
    ROI network that returns the locations and scores that doing roi pooling on the input proposals
    Args:
        classifier (Sequential Model): classifier from the backbone network
        in_features (Int): the number of features output by the classifier
        num_classes (Int): number of classes including background
        roi_size (Tuple[Int]): (Height, Width) of roi pooling
        spatial_scale (Float): scale of the input proposals to original input image
        
    """
    def __init__(self, 
                 backboneNetwork,
                 num_classes, 
                 roi_size, 
                 spatial_scale,
                 pos_prop_iou_thresh = 0.7, 
                 neg_prop_iou_thresh = 0.3, 
                 max_samples=256, 
                 ratio_pos_neg=0.5,
                 normalize_mean=(0.0,0.0,0.0,0.0),
                 normalize_std =(0.1,0.1,0.2,0.2)
                ):
        
        super(ROINetwork, self).__init__()
        
        self.num_classes = num_classes
        self.roi_height, self.roi_width = roi_size
        self.spatial_scale = spatial_scale
        #self.device = device if device != None else torch.device('cpu')
        self.backbone = backboneNetwork
        #Classifier Model
        self.classifier, self.in_features = self.backbone.getClassifier()
        #locations
        self.locations = torch.nn.Linear(self.in_features, self.num_classes * 4)
        #scores for the locations
        self.classes = torch.nn.Linear(self.in_features, self.num_classes)
        
        #ROI
        self.roi = torchvision.ops.RoIPool((self.roi_height, self.roi_width), self.spatial_scale)
        
        self.roiTrainGen = ROIProposalGenerator(num_classes, pos_prop_iou_thresh, neg_prop_iou_thresh, 
                                                max_samples, ratio_pos_neg,
                                                normalize_mean, normalize_std)
        
    
    def parseHead(self, input_):
        """
        Parse the given input into the roi classifier
        """
        out = self.classifier(input_)
        out_locs = self.locations(out)
        out_classes = self.classes(out)
        
        return out_locs, out_classes
    
    def classification_loss(self, pred_classes, gt_classes):
        #gt_classes = gt_classes.type_as(pred_classes)
        loss = torch.nn.functional.cross_entropy(pred_classes, gt_classes)
        return loss
    
    def regression_loss(self, pred_boxes, gt_boxes, gt_classes):
        gt_boxes = gt_boxes.type_as(pred_boxes)
        #Get the bounding boxes for the correct label
        
        loss = torch.nn.functional.smooth_l1_loss(pred_boxes, gt_boxes.view(-1, 4))
        
        return loss
    
    def parse(self, feat_maps, roi_proposals):
        """
        B -> Batch
        C -> Channels
        H -> Height
        W -> Widht
        N -> Number of proposals
        Args:
            feat_maps (B,C,H,W): the features map output from backbone network
            roi_proposals (Tensor (B, N, 4) or List[Tensor (N,4)]): region of interests
        """
        #Get batch size
        batch_size = feat_maps.size(dim=0)
        
        #Get the roi proposals in the list form depending input shape of roi_proposals
        if isinstance(roi_proposals, torch.Tensor):
            roi_props = [rois for rois in roi_proposals]
        elif isinstance(roi_proposals, list):
            roi_props = [rois for rois in roi_proposals]
        else:
            raise ValueError(f"Expected Tensor(B,N,4) or List[Tensors(N,4)] but got {type(roi_proposals)}")
        
        #get the pooled region of interests given, feature maps and region of interests
        roi_pool = self.roi(feat_maps, roi_props)
        roi_pool = self.backbone.roiResize(roi_pool)
        #parse pooled rois through the classifier
        out_locs, out_scores = self.parseHead(roi_pool)
        
        return out_locs, out_scores
    
    def forward(self, feat_maps, roi_proposals, gt_bboxes, gt_labels):
        """
        B -> Batch
        C -> Channels
        H -> Height
        W -> Widht
        N -> Number of proposals
        L -> Number of ground truth values
        Args:
            feat_maps (B,C,H,W): the features map output from backbone network
            roi_proposals (Tensor (B, N, 4) or List[Tensor (N,4)]): region of interests
            gt_bboxes (B,L,4): ground truth bounding boxes
            gt_labels (B,N,4): ground truth labels for each anchor
        """
        batch_size = gt_bboxes.size(dim=0)
        
        roi_props, gt_classes, gt_offsets = self.roiTrainGen(roi_proposals, gt_bboxes, gt_labels)
        
        out_locs, out_scores = self.parse(feat_maps, roi_props)
        
        gt_classes = gt_classes.view(-1)
        
        pred_boxes_per_gt_label = torch.gather(out_locs.view(out_locs.size(dim=0), -1, 4), 1, 
                                               gt_classes.view(gt_classes.size(dim=0), 1, 1).expand(gt_classes.size(dim=0), 1, 4))
        pred_boxes_per_gt_label = pred_boxes_per_gt_label.squeeze(1)
        
        
        locs_loss = self.regression_loss(pred_boxes_per_gt_label, gt_offsets, gt_classes)
        scores_loss = self.classification_loss(out_scores, gt_classes)
        
        scores_probs = torch.nn.functional.softmax(out_scores, 1)
        
        scores_probs = scores_probs.view(batch_size, roi_props.size(dim=1), -1)
        labels = torch.argmax(out_scores, 1).view(batch_size, -1)
        
        scores_probs_gathered = torch.gather(scores_probs.view(-1, self.num_classes), 1, labels.view(-1,1)).squeeze(1)
        
        roi_out = {
            "boxes": pred_boxes_per_gt_label.view(batch_size, -1, 4),
            "scores": scores_probs_gathered,
            "labels": labels,
            "rois": roi_props
        }
        
        roi_loss = {
            "roi_bbox_loss": locs_loss,
            "roi_cls_loss": scores_loss
        }
        
        
        return roi_loss, roi_out
    
    
    def inference(self, feat_maps, roi_proposals):
        with torch.no_grad():
            batch_size = feat_maps.size(dim=0)
            out_locs, out_scores = self.parse(feat_maps, roi_proposals)
            
            scores_probs = torch.nn.functional.softmax(out_scores, 1)
            
            scores_probs = scores_probs.view(batch_size, roi_proposals.size(dim=1), -1)
            labels = torch.argmax(out_scores, 1).view(batch_size, -1)
        
            scores_probs_gathered = torch.gather(scores_probs.view(-1, self.num_classes), 1, labels.view(-1,1)).squeeze(1).view(batch_size,-1)
            
            labels_ = labels.view(-1)
            
            pred_boxes = torch.gather(out_locs.view(out_locs.size(dim=0), -1, 4), 1, 
                                      labels_.view(labels_.size(dim=0), 1, 1).expand(labels_.size(dim=0), 1, 4))
            
            pred_boxes = pred_boxes.squeeze(1)
            
            roi_out = {
                "boxes": pred_boxes.view(batch_size, -1, 4),
                "scores": scores_probs_gathered,
                "labels": labels
            }
            
        return roi_out
    