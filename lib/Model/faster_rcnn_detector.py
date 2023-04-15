import torch.nn as nn
import torch
from lib.Model.backbones.vgg16 import VGG16_BACKBONE
from lib.Model.rpn_utils.rpn import RegionProposalNetwork
from lib.Model.roi_utils.roi import ROINetwork

class FRCNNDetector(nn.Module):
    def __init__(self, settings, device=None):
        
        super(FRCNNDetector, self).__init__()
        #self.cuda_device = "cuda" if device == None else device
        self.device = device if device != None else torch.device('cpu')
        self.settings = settings
        self.img_h, self.img_w = self.settings.get('IMG_HEIGHT'), self.settings.get('IMG_WIDTH')
        
        #Defining the model
        self.backbone = VGG16_BACKBONE(device=self.device)
        #ROI Classifier and in features
        self.classifier, self.roi_in_feats = self.backbone.getClassifier()
        
        #Get the fm_size
        self.fm_in_channels = self.settings.get('BACKBONE_OUT_CHANNELS')
        self.feat_scaler = self.settings.get('FEAT_SCALAR')
        self.fm_h = self.img_h / self.feat_scaler
        self.fm_w = self.img_w / self.feat_scaler
        
        self.rpn = RegionProposalNetwork(
            (self.fm_h,self.fm_w), 
            (self.img_h, self.img_w), 
            self.fm_in_channels,
            conf_score_weight = self.settings.get('CONF_LOSS_WEIGHT'), 
            bbox_weight = self.settings.get('BBOX_LOSS_WEIGHT'),
            pos_anchor_thresh = self.settings.get('POS_ANCHOR_THRESH'), 
            neg_anchor_thresh = self.settings.get('NEG_ANCHOR_THRESH'), 
            anc_ratio= self.settings.get('ANCHOR_RATIO'), 
            anchor_scales= self.settings.get('ANCHOR_SCALES'), 
            anchor_ratios = self.settings.get('ANCHOR_RATIOS'), 
            min_samples=self.settings.get('MIN_SAMPLES'), 
            stride= self.settings.get('STRIDE'), 
            device=self.device)
        
        self.roi = ROINetwork(self.classifier, self.roi_in_feats, 
                              num_classes=self.settings.get("NUM_CLASSES"),
                              roi_size=(self.settings.get("ROI_HEIGHT"),self.settings.get("ROI_WIDTH")),
                              spatial_scale = (1/self.feat_scaler),
                              device=self.device
                             )
        
    
    def forward(self, images, targets):
        out_feature_maps = self.backbone(images)
        
        rpn_loss, rpn_proposals, rpn_confs = self.rpn(out_feature_maps, targets['boxes'], targets['labels'])
        
        
        return rpn_loss
    
    def inference(self, images, nms_thresh=0.7, conf_thresh=0.5):
        feature_maps = self.backbone(images)
        
        proposals, conf_scores = self.rpn.inference(feature_maps)
        
        return proposals, conf_scores

