import torch
import torchvision

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
    def __init__(self, classifier, in_features, num_classes, roi_size, spatial_scale, device=None):
        super(ROINetwork, self).__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.roi_height, self.roi_width = roi_size
        self.spatial_scale = spatial_scale
        self.device = device if device != None else torch.device('cpu')
        
        #Classifier Model
        self.classifier = classifier.to(self.device)
        #locations
        self.locations = torch.nn.Linear(self.in_features, self.num_classes * 4, device=self.device)
        #scores for the locations
        self.scores = torch.nn.Linear(self.in_features, self.num_classes, device=self.device)
        
        #ROI
        self.roi = torchvision.ops.RoIPool((self.roi_height, self.roi_width), self.spatial_scale)
        
        
    
    def parseHead(self, input_):
        """
        Parse the given input into the roi classifier
        """
        out = self.classifier(input_)
        out_locs = self.locations(out)
        out_scores = self.scores(out)
        
        return out_locs, out_scores
    
    def classfication_loss(self, pred_labels, gt_labels):
        
        gt_labels = torch.nn.functional.one_hot(gt_labels.view(-1), num_classes).float()
        
        loss = torch.nn.functional.cross_entropy(pred_labels, gt_labels)
        return loss
    
    def regression_loss(self, pred_boxes, gt_boxes, gt_labels):
        
        #Get the bounding boxes for the correct label
        pred_boxes_per_gt_label = torch.mul(out_loc, gt_labels_.unsqueeze(-1)).sum(dim=1)
        
        loss = torch.nn.functional.smooth_l1_loss(pred_boxes_per_gt_label, gt_boxes)
        
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
            roi_props = [rois.to(self.device) for rois in roi_proposals]
        elif isinstance(roi_proposals, list):
            roi_props = [rois.to(self.device) for rois in roi_proposals]
        else:
            raise ValueError(f"Expected Tensor(B,N,4) or List[Tensors(N,4)] but got {type(roi_proposals)}")
        
        #get the pooled region of interests given, feature maps and region of interests
        roi_pool = self.roi(feat_maps, roi_props)
        roi_pool = roi_pool.to(self.device)
        roi_pool = roi_pool.view(roi_pool.size(dim=0), -1)
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
        out_locs, out_scores = self.parse(feat_maps, roi_proposals)
        
        #
        
        
        return out_locs, out_scores
    
    
    def inference(self, feat_maps, roi_proposals):
        with torch.no_grad():
            out_locs, out_scores = self.parse(feat_maps, roi_proposals)
            
        return out_locs, out_scores
    