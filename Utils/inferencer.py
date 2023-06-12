from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.utils import crop_img, poly2bbox, bbox2poly
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


class Inferencer:
    def __init__(self, det_model="DBnet", det_weights=None, rec_model="satrn", rec_weights=None, output_dir="./inference/", print_vis=True, device=torch.device("cuda")):
        
        if (isinstance(det_model, str)):
            assert det_model in ["DBnet", "MaskRCNN"], "Expected detection model to be either custom model or one of the following: DBnet, MaskRCNN"
        
        assert rec_model in ["crnn", "satrn"], "Current Inferencer only supports crnn and satrn recognition models"
        
        self.det_model = self.getDetModel(det_model, det_weights, device)
        self.rec_model = self.getRecModel(rec_model, rec_weights, device)
        self.output_dir = output_dir
        self.print_vis = print_vis
        self.device = device
    
    
    def getDetModel(self, det, weights, device):     
        
        if isinstance(det, str):
            model = TextDetInferencer(det, weights, device)
        else:
            model = det
            
            if weights != None:
                model = model.load_state_dict(weights)
            
        
        return model
    
    def getRecModel(self, rec, weights, device):
        
        model = TextRecInferencer(rec, weights, device)
        
        return model
    
    def process_det_ocr_out(self, out):
        
        preds = out["predictions"]
        
        out_list = []
        
        for inst in preds:
            poly = inst["polygons"]
            
            quad = bbox2poly(poly2bbox(poly)).tolist()
            
            scores = inst["scores"]
            
            final_dict = dict(boxes=quad, scores=scores)
            
            out_list.append(final_dict)
        
        return out_list
    
    def process_det_out(self, out):
        
        if not("boxes" in out[0].keys()):
            out = out[1]
        
        out_list = []
        
        for inst in out:
            final_dict = dict(boxes=bbox2poly(out['boxes']), scores=out["scores"])
            out_list.append(final_dict)
        
        return out_list
    
    def save_vis(self, images, outs):
        
        try:
            os.makedirs(self.output_dir)
        except:
            pass
        
        for index, (image, out) in enumerate(zip(images, outs)):
            h, w = image.shape[-2:]
            
            img = Image.new("RGB", (w+w, h))
            
            img.paste(image, (0,0))
            img.paste(image, (w,0))
            
            fig, ax = plt.subplots(figsize=(8,8))
            plt.axis("off")
            
            plt.imshow(img)
            
            for index, box in enumerate(out["boxes"]):
                box_score = out["scores"][index]
                text = out["text"][index]
                text_score = out["text_score"][index]
                
                x1,y1,_,_,x3,y3,_,_ = box
                
                width = x3-x1
                height = y3-y1
                
                box_color = "red" if box_score < 0.5 else "green"
                text_color = "red" if text_score < 0.5 else "green"
                
                rect1 = patches.Rectangle((x1,y1), width, height, fill=None, fc=box_color, ec=box_color)
                rect2 = patches.Rectangle((w+x1,y1), width, height, fill=None, fc=box_color, ec=box_color)
                
                
                ax.add_patch(rect1)
                ax.add_patch(rect2)
            
                ax.text(w+x1, y1, text, color=text_color, va="bottom", ha="center")
            
            plt.save_fig(self.output_dir+f"image_{index}.jpg")
            
            plt.show()
        
        
    def __call__(self, image):
        
        if not(isinstance(image, list)):
            images = [image]
        else:
            images = image
            
        if isinstance(self.det_model, TextDetInferencer):
            out = self.det_model(images)
            out = self.process_det_ocr_out(out)
        else:
            out = self.det_model(images)
            out = self.process_det_out(out)
            
            
        for index,(image, preds) in enumerate(zip(images, out)):
            cropped_proposals = []
            
            for box in pred["boxes"]:
                cropped_proposals.append(crop_img(image, box))
            
            preds = self.rec_model(cropped_proposals)["predictions"]
            
            all_text = [pred["text"] for pred in preds]
            all_scores = [pred["scores"] for pred in preds]
            
            out[index]["text"] = all_text
            out[index]["text_scores"] = all_scores
        
        
        
        if self.print_vis:
            self.save_vis(images, out)
            
        print(out)
        return out
        
        