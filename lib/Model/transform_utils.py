import torch
import copy

"""
A data transformer, that makes sure the input image and/or training data is within the dimensions of min_size and max_size
Transformations are also done on the input data to make sure of the previous statement
"""
class FRCNNTransformer(torch.nn.Module):
    
    def __init__(self, min_size, max_size, img_mean, img_std):
        super(FRCNNTransformer, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.img_mean = img_mean
        self.img_std = img_std

    """
    Given an image, resize the image such that the longest side is within the max_size
    """
    def resize_image(self, image):

        #Get the original dimensions
        c, h, w = image.shape

        #Calcualte scale factor wrt minimum size
        scale_factor = float(self.min_size) / float(min(h,w))

        #Check if scale factor minimises the size of the longest side
        #If not then calculate scale factor wrt to maximum size
        if scale_factor * float(max(h,w)) > self.max_size:
            scale_factor = float(self.max_size) / float(max(h,w))

        #Resize the image
        image = torch.nn.functional.interpolate(image.view(1, c, h, w), scale_factor=scale_factor, mode='bilinear')[0]

        return image
        

    """
    Given a list of boxes, current shape and final shape
    Resize the boxes such that they are boxes for image of final shape
    """
    def resize_boxes(self, boxes, current_shape, to_shape):

        #Get the dimensions of the shapes
        _, current_h, current_w = current_shape
        _, to_h, to_w = to_shape

        #Calculate the ratio
        h_ratio = to_h / current_h 
        w_ratio = to_w / current_w

        #Perform the resizing given ratio
        boxes[...,[0,2]] *= w_ratio
        boxes[...,[1,3]] *= h_ratio
        
        return boxes

    def forward(self, images, targets=None):

        #Final output
        final_images = []
        final_targets = copy.deepcopy(targets) if not(targets is None) else None
        #Original image size list
        images_orig_size = [image.shape for image in images]

        #Iterate through all available images and perform transformation on images and target data if available
        for index, image in enumerate(images):
            #Resize Image
            image = self.resize_image(image)

            #Normalising image using: Mean subtraction and Std division to improve performance and convergence
            image = (image - torch.as_tensor(self.img_mean).type_as(image)[:, None, None]) / torch.as_tensor(self.img_std).type_as(image)[:, None, None]

            #Get new shape and old shape
            new_shape = image.shape
            orig_shape = images_orig_size[index]

            #Resize boxes
            if not(targets is None):
                boxes = final_targets[index]['boxes']
                final_targets[index]['boxes'] = self.resize_boxes(boxes, orig_shape, new_shape)
                
            final_images.append(image)
        
        
        return final_images, final_targets, images_orig_size
    
            
        
        
