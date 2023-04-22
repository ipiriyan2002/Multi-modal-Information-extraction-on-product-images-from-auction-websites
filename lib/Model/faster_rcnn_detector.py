import torch.nn as nn
import torchvision.ops as ops
import torch
from lib.Model.backbones.vgg16 import VGG16_BACKBONE
from lib.Model.rpn_utils.rpn import RegionProposalNetwork
from lib.Model.roi_utils.roi import ROINetwork

class FRCNNDetector(nn.Module):
    def __init__(self, settings):
        
        super(FRCNNDetector, self).__init__()
        #self.cuda_device = "cuda" if device == None else device
        self.settings = settings
        self.img_h, self.img_w = self.settings.get('IMG_HEIGHT'), self.settings.get('IMG_WIDTH')
        
        #Defining the model
        self.backboneNetwork = VGG16_BACKBONE(settings.get("PRETRAINED"), settings.get("PRETRAINED_PATH"))
        self.backbone = self.backboneNetwork.getModel()
        
        #Get the fm_size
        self.fm_in_channels = self.settings.get("BACKBONE_OUT_CHANNELS")
        self.feat_scaler = self.settings.get('FEAT_SCALAR')
        self.fm_h = self.img_h / self.feat_scaler
        self.fm_w = self.img_w / self.feat_scaler
        
        self.rpn = RegionProposalNetwork(
            (self.fm_h,self.fm_w), 
            (self.img_h, self.img_w), 
            self.fm_in_channels,
            conf_score_weight=self.settings.get('CONF_LOSS_WEIGHT'), 
            bbox_weight=self.settings.get('BBOX_LOSS_WEIGHT'),
            pos_anchor_thresh=self.settings.get('POS_ANCHOR_THRESH'), 
            neg_anchor_thresh=self.settings.get('NEG_ANCHOR_THRESH'), 
            anc_ratio=self.settings.get('ANCHOR_RATIO'), 
            anchor_scales=self.settings.get('ANCHOR_SCALES'), 
            anchor_ratios=self.settings.get('ANCHOR_RATIOS'), 
            max_samples=self.settings.get('RPN_MAX_SAMPLES'), 
            stride=self.settings.get('STRIDE'))
        
        self.roi = ROINetwork(self.backboneNetwork,
                              num_classes=self.settings.get("NUM_CLASSES"),
                              roi_size=(self.settings.get("ROI_HEIGHT"),self.settings.get("ROI_WIDTH")),
                              spatial_scale = (1/self.feat_scaler),
                              pos_prop_iou_thresh = self.settings.get("POS_PROP_THRESH"), 
                              neg_prop_iou_thresh = self.settings.get("NEG_PROP_THRESH"), 
                              max_samples=self.settings.get("ROI_MAX_SAMPLES"), 
                              ratio_pos_neg=self.settings.get("ROI_RATIO"),
                              normalize_mean=self.settings.get("NORMALIZE_MEAN"),
                              normalize_std =self.settings.get("NORMALIZE_STD")
                             )
        
    def generateProposals(self, box_deltas, boxes):
        boxes = ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')
        
        x_ctr = boxes[...,0] + (box_deltas[...,0] * boxes[...,2])
        y_ctr = boxes[...,1] + (box_deltas[...,1] * boxes[...,3])
        width = torch.exp(box_deltas[...,2]) * boxes[...,2]
        height = torch.exp(box_deltas[...,3]) * boxes[...,3]
        
        props = torch.stack([x_ctr, y_ctr, width, height], axis=2)
        
        return ops.box_convert(props, in_fmt='cxcywh', out_fmt='xyxy')
    
    def forward(self, images, targets):
        out_feature_maps = self.backbone(images)
        
        rpn_loss, rpn_out = self.rpn(out_feature_maps, targets['boxes'], targets['labels'])
        
        roi_loss, roi_out = self.roi(out_feature_maps, rpn_out['proposals'], targets['boxes'], targets['labels'])
        
        roi_out['boxes'] = self.generateProposals(roi_out['boxes'], roi_out['rois'])
        
        #Get all losses into one loss dict
        loss_dict = {}
        for losses in [rpn_loss,roi_loss]:
            for k,v in losses.items():
                loss_dict[k] = v
        
        return loss_dict, roi_out
    
    def inference(self, images, nms_thresh=0.7, conf_thresh=0.5):
        with torch.no_grad():
        
            feature_maps = self.backbone(images)

            rpn_out = self.rpn.inference(feature_maps)

            roi_out = self.roi.inference(feature_maps, rpn_out['proposals'])
            
            boxes = self.generateProposals(roi_out['boxes'], rpn_out['proposals'])
            labels = roi_out['labels']
            scores = roi_out['scores']
            
            #Remove all predictions with score above threshold
            in_confs = [torch.where(score >= conf_thresh)[0] for score in scores]
            
            boxes = [box[in_conf] for box,in_conf in zip(boxes, in_confs)]
            labels = [label[in_conf] for label,in_conf in zip(labels, in_confs)]
            scores= [score[in_conf] for score,in_conf in zip(scores, in_confs)]
            
            #Perform non-maximum suppression
            nms_pos = [ops.nms(box,score,nms_thresh) for box,score in zip(boxes,scores)]
            
            boxes = [box[nms] for box, nms in zip(boxes,nms_pos)]
            labels = [label[nms] for label, nms in zip(labels,nms_pos)]
            scores = [score[nms] for score, nms in zip(scores,nms_pos)]
            
            frcnn_out = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            }
        
        return frcnn_out

