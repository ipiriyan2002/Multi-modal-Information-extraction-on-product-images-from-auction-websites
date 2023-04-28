import torchvision
import random
import torch

class FRCNNTransformer(torch.nn.Module):
    
    def __init__(self, min_size, max_size, img_mean, img_std):
        super(FRCNNTransformer, self).__init__()

        self.min_size = min_size
        self.max_size = max_size
        self.img_mean = img_mean
        self.img_std = img_std
    
    def resize_image(self, image):
        
        c, h, w = image.shape
        scale_factor = float(self.min_size) / float(torch.min(torch.tensor(image.shape[-2:])))
        
        if scale_factor * float(torch.max(torch.tensor(image.shape[-2:]))) > self.max_size:
            scale_factor = float(self.max_size) / float(torch.max(torch.tensor(image.shape[-2:])))
        
        image = torch.nn.functional.interpolate(image.view(1, c, h, w), scale_factor=scale_factor, mode='bilinear')[0]
        
        
        return image
        
    
    def resize_boxes(self, boxes, current_shape, to_shape):
        
        _, current_h, current_w = current_shape
        _, to_h, to_w = to_shape
        
        h_ratio = current_h / to_h
        w_ratio = current_w / to_w
        
        boxes[...,[0,2]] = boxes[...,[0,2]] * w_ratio
        boxes[...,[1,3]] = boxes[...,[1,3]] * h_ratio
        
        return boxes
    
    def forward(self, images, targets=None):
        
        final_images = []
        images_orig_size = [image.shape for image in images]
        
        for index, image in enumerate(images):
            #Normalising image using:
            #1) Mean subtraction and Std division to improve performance and convergence
            
            image = (image - torch.as_tensor(self.img_mean).type_as(image)[:, None, None]) / torch.as_tensor(self.img_std).type_as(image)[:, None, None]
            
            image = self.resize_image(image)
            
            new_shape = image.shape
            orig_shape = images_orig_size[index]
            
            if targets != None:
                targets[index]['boxes'] = self.resize_boxes(targets[index]['boxes'], orig_shape, new_shape)
                
            final_images.append(image)
        
        
        return final_images, targets, images_orig_size
    
            
        
        