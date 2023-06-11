import torch.nn as nn
import torchvision.ops as ops
import torch
from lib.Model.backbones.vgg16 import VGG16_BACKBONE
from lib.Model.rpn_utils.rpn import RegionProposalNetwork
from lib.Model.roi_utils.roi import ROINetwork
from lib.Model.transform_utils import FRCNNTransformer

class FRCNNDetector(nn.Module):
    def __init__(self, settings):
        
        super(FRCNNDetector, self).__init__()
        self.settings = settings
        self.min_size, self.max_size = self.settings.get('MIN_SIZE'), self.settings.get('MAX_SIZE')

        #Define the transformer
        self.transform = FRCNNTransformer(
            min_size=self.min_size,
            max_size=self.max_size,
            img_mean=self.settings.get("IMAGE_MEAN"),
            img_std=self.settings.get("IMAGE_STD")
        )
        
        #Defining the backbone
        self.backboneNetwork = VGG16_BACKBONE(settings.get("PRETRAINED"), settings.get("PRETRAINED_PATH"))
        self.backbone = self.backboneNetwork.getModel()
        
        #Get the fm_size
        self.fm_in_channels = self.backboneNetwork.getOutChannels()
        self.feat_scaler = self.backboneNetwork.getFeatScaler()
        
        #Define the rpn
        self.rpn_pre_nms_k = {
            "TRAIN": self.settings.get('TRAIN_RPN_PRE_NMS_K'),
            "TEST": self.settings.get('TEST_RPN_PRE_NMS_K')
        }
        self.rpn_post_nms_k = {
            "TRAIN": self.settings.get('TRAIN_RPN_POST_NMS_K'),
            "TEST": self.settings.get('TEST_RPN_POST_NMS_K')
        }
        self.rpn = RegionProposalNetwork(
            rpn_in_channels=self.fm_in_channels,
            conf_score_weight=self.settings.get('CONF_LOSS_WEIGHT'), 
            bbox_weight=self.settings.get('BBOX_LOSS_WEIGHT'),
            pos_anchor_thresh=self.settings.get('POS_ANCHOR_THRESH'), 
            neg_anchor_thresh=self.settings.get('NEG_ANCHOR_THRESH'), 
            anc_ratio=self.settings.get('ANCHOR_RATIO'), 
            anchor_scales=self.settings.get('ANCHOR_SCALES'), 
            anchor_ratios=self.settings.get('ANCHOR_RATIOS'), 
            max_samples=self.settings.get('RPN_MAX_SAMPLES'), 
            stride=self.feat_scaler,
            img_max_size=self.max_size,
            conf_thresh=self.settings.get('RPN_CONF_THRESH'),
            nms_thresh=self.settings.get('RPN_NMS_THRESH'),
            pre_nms_k=self.rpn_pre_nms_k,
            post_nms_k=self.rpn_post_nms_k
        )
        
        #Define the roi network
        self.roi = ROINetwork(self.backboneNetwork,
                              num_classes=self.settings.get("NUM_CLASSES"),
                              roi_size=(self.settings.get("ROI_HEIGHT"),self.settings.get("ROI_WIDTH")),
                              spatial_scale = (1/self.feat_scaler),
                              pos_prop_iou_thresh = self.settings.get("POS_PROP_THRESH"), 
                              neg_prop_iou_thresh = self.settings.get("NEG_PROP_THRESH"), 
                              max_samples=self.settings.get("ROI_MAX_SAMPLES"), 
                              ratio_pos_neg=self.settings.get("ROI_RATIO"),
                              roi_post_nms_k = self.settings.get("TEST_POST_K"),
                              roi_conf_thresh = self.settings.get("ROI_CONF_THRESH"),
                              roi_nms_thresh = self.settings.get("ROI_NMS_THRESH"),
                              weights=self.settings.get("ROI_WEIGHTS")
                             )
    
    def forward(self, images, targets=None):
        """
        Return the losses and detections if training
        else detections
        """
        if self.training:
            assert (targets != None), "Expected targets for training"

        #Perform data transformation and retreive original size of image
        images, targets, orig_img_sizes = self.transform(images, targets)

        #Get all the feature maps for each image
        feature_maps = [self.backbone(image) for image in images]

        #Get the RPN outputs
        rpn_loss, rpn_out = self.rpn(feature_maps, images, targets)

        #Get the ROI outputs
        roi_loss, roi_out = self.roi(feature_maps, rpn_out['proposals'], images, orig_img_sizes, targets)

        #Get all losses into one loss dict
        loss_dict = {}
        loss_dict.update(rpn_loss)
        loss_dict.update(roi_loss)

        return loss_dict, roi_out

