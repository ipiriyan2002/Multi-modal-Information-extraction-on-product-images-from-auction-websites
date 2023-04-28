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
        #self.cuda_device = "cuda" if device == None else device
        self.settings = settings
        self.min_size, self.max_size = self.settings.get('MIN_SIZE'), self.settings.get('MAX_SIZE')
        
        #Defining the model
        self.backboneNetwork = VGG16_BACKBONE(settings.get("PRETRAINED"), settings.get("PRETRAINED_PATH"))
        self.backbone = self.backboneNetwork.getModel()
        
        #Get the fm_size
        self.fm_in_channels = self.settings.get("BACKBONE_OUT_CHANNELS")
        self.feat_scaler = self.settings.get('FEAT_SCALAR')
        
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
            self.fm_in_channels,
            conf_score_weight=self.settings.get('CONF_LOSS_WEIGHT'), 
            bbox_weight=self.settings.get('BBOX_LOSS_WEIGHT'),
            pos_anchor_thresh=self.settings.get('POS_ANCHOR_THRESH'), 
            neg_anchor_thresh=self.settings.get('NEG_ANCHOR_THRESH'), 
            anc_ratio=self.settings.get('ANCHOR_RATIO'), 
            anchor_scales=self.settings.get('ANCHOR_SCALES'), 
            anchor_ratios=self.settings.get('ANCHOR_RATIOS'), 
            max_samples=self.settings.get('RPN_MAX_SAMPLES'), 
            stride=self.settings.get('STRIDE'),
            score_thresh=self.settings.get('RPN_CONF_THRESH'),
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
                              ratio_pos_neg=self.settings.get("ROI_RATIO")
                             )
        self.roi_post_nms_k = self.settings.get("TEST_POST_K")
        self.roi_conf_thresh = self.settings.get("ROI_CONF_THRESH")
        self.roi_nms_thresh = self.settings.get("ROI_NMS_THRESH")
        
        #Define the transformer
        self.transform = FRCNNTransformer(self.min_size, self.max_size, 
                                          img_mean=self.settings.get("IMAGE_MEAN"), 
                                          img_std=self.settings.get("IMAGE_STD"))
        
    def generateProposals(self, box_deltas, boxes):
        """
        Generate proposals given the box offsets and boxes to apply the offsets on
        """
        if not(isinstance(boxes, torch.Tensor)):
            boxes = torch.cat(boxes, dim=0)
        
        boxes = ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')
        
        x_ctr = boxes[...,0] + (box_deltas[...,0] * boxes[...,2])
        y_ctr = boxes[...,1] + (box_deltas[...,1] * boxes[...,3])
        width = torch.exp(box_deltas[...,2]) * boxes[...,2]
        height = torch.exp(box_deltas[...,3]) * boxes[...,3]
        
        props = torch.stack([x_ctr, y_ctr, width, height], axis=2)
        
        return ops.box_convert(props, in_fmt='cxcywh', out_fmt='xyxy')
    
    def post_process(self, boxes, scores, labels, img_shapes):
        
        #Clip boxes to image
        boxes = [ops.clip_boxes_to_image(box, img_shape[-2:]) for box, img_shape in zip(boxes, img_shapes)]
        
        #remove small boxes
        keeps = [ops.remove_small_boxes(box, 1e-3) for box in boxes]
        boxes = [box[keep] for box, keep in zip(boxes, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        labels = [label[keep] for label, keep in zip(labels, keeps)]
        
        #remove boxes with low score
        keeps = [torch.where(score >= self.roi_conf_thresh)[0] for score in scores]
        boxes = [box[keep] for box, keep in zip(boxes, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        labels = [label[keep] for label, keep in zip(labels, keeps)]
        
        #perform nms
        keeps = [ops.nms(box,score,self.roi_nms_thresh) for box,score in zip(boxes,scores)]
        boxes = [box[keep] for box, keep in zip(boxes, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        labels = [label[keep] for label, keep in zip(labels, keeps)]
        
        if not(self.training):
            boxes = [box[:self.roi_post_nms_k] for box in boxes]
            scores = [score[:self.roi_post_nms_k] for score in scores]
            labels = [label[:self.roi_post_nms_k] for label in labels]
        
        return boxes, scores, labels
    
    def forward(self, images, targets=None):
        """
        Return the losses and detections if training
        else detections
        """
        
        images, targets, orig_img_sizes = self.transform(images, targets)
        
        if self.training:
            assert (targets != None), "Expected targets for training"
            parse_images = torch.cat(images, dim=0)
            
            if len(parse_images.size()) == 3:
                c, h, w = parse_images.shape
                parse_images = parse_images.view(1, c, h, w)
            
            feature_maps = self.backbone(parse_images)

            rpn_loss, rpn_out = self.rpn(feature_maps, images, targets)

            roi_loss, roi_out = self.roi(feature_maps, rpn_out['proposals'], targets)

            boxes = self.generateProposals(roi_out['boxes'].detach(), roi_out['rois'].detach())
            boxes = self.resize_boxes(images, orig_img_sizes, boxes)
            
            boxes, scores, labels = self.post_process(boxes, roi_out['scores'], roi_out['labels'], orig_img_sizes)
            
            frcnn_out = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            }
            
            #Get all losses into one loss dict
            loss_dict = {}
            loss_dict.update(rpn_loss)
            loss_dict.update(roi_loss)

            return loss_dict, frcnn_out
        else:
            return self.inference(images, orig_img_sizes)
    
    def resize_boxes(self, images, image_sizes, all_boxes):
        
        for index, boxes in enumerate(all_boxes):
            
            _, current_h, current_w = images[index].shape
            _, to_h, to_w = image_sizes[index]
            
            h_ratios = current_h / to_h
            w_ratios = current_w / to_w
            
            boxes[...,[0,2]] = boxes[...,[0,2]] * w_ratios
            boxes[...,[1,3]] = boxes[...,[1,3]] * h_ratios
        
        return all_boxes
    
    def inference(self, images, orig_img_sizes):
        with torch.no_grad():
            parse_images = torch.cat(images, dim=0)
            
            if len(parse_images.size()) == 3:
                c, h, w = parse_images.shape
                parse_images = parse_images.view(1, c, h, w)
                batch_size = 1
            elif len(parse_images.size()) == 4:
                batch_size = parse_images.shape[0]
            else:
                raise ValueError("Invalid image shape, expected either BxCxHxW or CxHxW")
                
            feature_maps = self.backbone(parse_images)

            rpn_out = self.rpn(feature_maps, images)

            roi_out = self.roi(feature_maps, rpn_out['proposals'])
            
            boxes = self.generateProposals(roi_out['boxes'], rpn_out['proposals'])
            labels = roi_out['labels']
            scores = roi_out['scores']
            
            boxes = self.resize_boxes(images, orig_img_sizes, boxes)
            
            boxes, scores, labels = self.post_process(boxes, scores, labels, orig_img_sizes)
            
            frcnn_out = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            }
        
        return frcnn_out

