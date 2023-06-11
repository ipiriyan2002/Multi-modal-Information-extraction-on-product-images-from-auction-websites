import torch
import torchvision
from lib.Model.roi_utils.roi_proposal_gen import ROIProposalGenerator
import torchvision.ops as ops
import math

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
                 img_max_size=1000,
                 pos_prop_iou_thresh = 0.7, 
                 neg_prop_iou_thresh = 0.3, 
                 max_samples=128,
                 ratio_pos_neg=0.25,
                 roi_post_nms_k= 200,
                 roi_conf_thresh=0.05,
                 roi_nms_thresh=0.5,
                 weights=(1.0,1.0,1.0,1.0)
                ):
        
        super(ROINetwork, self).__init__()
        
        self.num_classes = num_classes
        self.roi_height, self.roi_width = roi_size
        self.spatial_scale = spatial_scale
        self.weights = weights
        self.roi_post_nms_k = roi_post_nms_k
        self.roi_conf_thresh = roi_conf_thresh
        self.roi_nms_thresh = roi_nms_thresh
        self.backbone = backboneNetwork
        #ROI
        self.roi = torchvision.ops.RoIPool((self.roi_height, self.roi_width), self.spatial_scale)

        #Using clamp size to limit the input width and height delta into exponential as higher deltas than an expected limit is cauing problems
        self.clamp_size = math.log(img_max_size * self.spatial_scale)
        self.roiTrainGen = ROIProposalGenerator(num_classes, pos_prop_iou_thresh, neg_prop_iou_thresh, 
                                                max_samples, ratio_pos_neg)
        #Classifier Model
        self.classifier, self.in_features = self.backbone.getClassifier()
        
        #locations
        self.locations = torch.nn.Linear(self.in_features, self.num_classes * 4)
        #scores for the locations
        self.classes = torch.nn.Linear(self.in_features, self.num_classes)
        
        self.init_weights(self.locations, 0, 0.001)
        self.init_weights(self.classes, 0, 0.01)
        
    
    def init_weights(self, layer, mean, std):
        layer.weight.data.normal_(mean, std)
        layer.bias.data.zero_()    
    
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
        loss = torch.nn.functional.cross_entropy(pred_classes, gt_classes.view(-1))
        return loss
    
    def regression_loss(self, pred_boxes, gt_boxes, gt_classes):
        
        batch_size = gt_classes.shape[0]
        num_gt_proposals = gt_classes.shape[1]
        
        gt_boxes = gt_boxes.type_as(pred_boxes).reshape(-1, 4)
        pred_boxes = pred_boxes.reshape(-1,self.num_classes, 4)

        sampled_indices = torch.where(gt_classes.view(-1) > 0)[0]
        pos = gt_classes.view(-1)[sampled_indices]
        
        loss = torch.nn.functional.smooth_l1_loss(pred_boxes[sampled_indices,pos], gt_boxes[sampled_indices], reduction='sum', beta=1/9)
        loss = loss / gt_classes.view(-1).numel()
        
        return loss
    
    def parse(self, feat_maps, roi_proposals):
        """
        B -> Batch
        C -> Channels
        H -> Height
        W -> Widht
        N -> Number of proposals
        Args:
            feat_maps List[Tensor[(C,H,W)]]: the features map output from backbone network
            roi_proposals (Tensor (B, N, 4) or List[Tensor (N,4)]): region of interests
        """
        #Get batch size
        batch_size = len(feat_maps)
        
        #Get the roi proposals in the list form depending input shape of roi_proposals
        if isinstance(roi_proposals, torch.Tensor):
            roi_props = [rois for rois in roi_proposals]
        elif isinstance(roi_proposals, list):
            roi_props = [rois for rois in roi_proposals]
        else:
            raise ValueError(f"Expected Tensor(B,N,4) or List[Tensors(N,4)] but got {type(roi_proposals)}")
        
        #get the pooled region of interests given, feature maps and region of interests
        out_locs = []
        out_scores = []
        for feat_map, roi_prop in zip(feat_maps, roi_props):
            c,h,w = feat_map.shape
            roi_pool = self.roi(feat_map.view(1,c,h,w), [roi_prop])
            roi_pool = self.backbone.roiResize(roi_pool)
            #parse pooled rois through the classifier
            out_loc, out_score = self.parseHead(roi_pool)
            out_locs.append(out_loc)
            out_scores.append(out_score)
        
        return torch.cat(out_locs, dim=0).view(-1, self.num_classes*4), torch.cat(out_scores, dim=0).view(-1, self.num_classes)
    
    def generateProposals(self, box_deltas, boxes):
        """
        Generate proposals given the box offsets and boxes to apply the offsets on
        """
        try:
            boxes = boxes.detach()
        except:
            pass

        #dividing by weights to make sure the offsets predicted are not affected by the weights when proposing the final rois
        box_deltas = box_deltas.reshape(boxes.size(0),-1)
        box_deltas[:,0::4] /= self.weights[0]
        box_deltas[:,1::4] /= self.weights[1]
        box_deltas[:,2::4] /= self.weights[2]
        box_deltas[:,3::4] /= self.weights[3]

        boxes = ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')
        
        
        x_ctr = boxes[:,0,None] + (box_deltas[:,0::4] * boxes[:,2,None])
        y_ctr = boxes[:,1,None] + (box_deltas[:,1::4] * boxes[:,3,None])
        width = torch.exp(torch.clamp(box_deltas[:,2::4], max=self.clamp_size)) * boxes[:,2,None]
        height = torch.exp(torch.clamp(box_deltas[:,3::4], max=self.clamp_size)) * boxes[:,3,None]
        
        props = torch.stack([x_ctr, y_ctr, width, height], axis=2).view(-1, 4)  
        
        return ops.box_convert(props, in_fmt='cxcywh', out_fmt='xyxy')
    
    def post_process(self, boxes, scores, labels, images):
        
        
        #Clip boxes to image
        boxes = [ops.clip_boxes_to_image(box, img.shape[-2:]) for box, img in zip(boxes, images)]
        
        #Make sure each element of boxes,scores and labels are flattened
        boxes = [box.reshape(-1,4) for box in boxes]
        scores = [score.reshape(-1) for score in scores]
        labels = [label.reshape(-1) for label in labels]
        
        #remove background proposals
        keeps = [torch.where(label > 0)[0] for label in labels]
        boxes = [box[keep] for box, keep in zip(boxes, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        labels = [label[keep] for label, keep in zip(labels, keeps)]
        
        #remove small boxes
        keeps = [ops.remove_small_boxes(box, 1e-3) for box in boxes]
        boxes = [box[keep] for box, keep in zip(boxes, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        labels = [label[keep] for label, keep in zip(labels, keeps)]
        
        #remove boxes with low score
        keeps = [torch.where(score > self.roi_conf_thresh)[0] for score in scores]
        boxes = [box[keep] for box, keep in zip(boxes, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        labels = [label[keep] for label, keep in zip(labels, keeps)]
        
        #perform nms
        keeps = [ops.batched_nms(box,score,label, self.roi_nms_thresh) for box,score,label in zip(boxes,scores,labels)]
        boxes = [box[keep] for box, keep in zip(boxes, keeps)]
        scores = [score[keep] for score, keep in zip(scores, keeps)]
        labels = [label[keep] for label, keep in zip(labels, keeps)]
        
        if not(self.training):
            boxes = [box[:self.roi_post_nms_k] for box in boxes]
            scores = [score[:self.roi_post_nms_k] for score in scores]
            labels = [label[:self.roi_post_nms_k] for label in labels]
        
        return boxes, scores, labels
    
    def resize_boxes(self, feat_maps, image_sizes, all_boxes):
        
        for index, boxes in enumerate(all_boxes):
            
            fm_h, fm_w = feat_maps[index].shape[-2:]
            current_h, current_w = (fm_h / self.spatial_scale), (fm_w / self.spatial_scale)
            _, to_h, to_w = image_sizes[index]
            
            h_ratios = to_h / current_h 
            w_ratios = to_w / current_w
            
            boxes[...,[0,2]] = boxes[...,[0,2]] * w_ratios
            boxes[...,[1,3]] = boxes[...,[1,3]] * h_ratios
        
        return all_boxes
    
    def forward(self, feat_maps, roi_proposals, images, orig_img_sizes, targets=None):
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
            targets: ground truth targets
        """
        
        batch_size = len(feat_maps)
        
        if self.training:
            assert (targets != None), "Expected targets for training"
            
            gt_bboxes = [target['boxes'] for target in targets]
            gt_labels = [target['labels'] for target in targets]
            
            roi_props, gt_classes, gt_offsets = self.roiTrainGen(roi_proposals, gt_bboxes, gt_labels)

        else:
            roi_props = torch.stack(roi_proposals, dim=0)
            gt_classes = None
            gt_offsets = None   
        
        
        out_locs, out_scores = self.parse(feat_maps, roi_props)

        roi_out = []

        if self.training:
            locs_loss = self.regression_loss(out_locs, gt_offsets, gt_classes)
            scores_loss = self.classification_loss(out_scores, gt_classes)
        else:
            locs_loss = None
            scores_loss = None

            #Probabilities
            scores_probs = torch.nn.functional.softmax(out_scores, -1)
            scores_probs = scores_probs.view(-1,self.num_classes)
            #all labels to be processed
            labels = torch.arange(self.num_classes).type_as(scores_probs).view(1,self.num_classes).expand_as(scores_probs).view(-1,self.num_classes)

            #Each image has a different number of proposals
            proposals_per_image = [props.size(0) * self.num_classes for props in roi_props]

            boxes = self.generateProposals(out_locs.detach(), roi_props.view(-1,4))
            boxes = boxes.split(proposals_per_image, 0)

            scores = scores_probs.flatten().split(proposals_per_image, 0)
            labels = labels.flatten().split(proposals_per_image, 0)

            boxes, scores, labels = self.post_process(boxes, scores, labels, images)
            boxes = self.resize_boxes(feat_maps, orig_img_sizes, boxes)


            for index in range(batch_size):
                dict_ = {
                    "boxes":boxes[index],
                    "scores":scores[index],
                    "labels":labels[index],
                    "rois": roi_props[index]
                }

                roi_out.append(dict_)
            
        roi_loss = {
            "roi_box_loss": locs_loss,
            "roi_cls_loss": scores_loss
        }
        
        
        return roi_loss, roi_out