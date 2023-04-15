import torch
import torch.nn as nn
import torchvision

class VGG16_BACKBONE(nn.Module):
    """
    VGG16 backbone network and also their classifier
    """
    def __init__(self, device=None):
        super(VGG16_BACKBONE, self).__init__()
        self.device = device if device != None else torch.device('cpu')
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16 = self.vgg16.to(self.device)
        self.vgg16_features = list(self.vgg16.features)
        
        self.bbNet = nn.Sequential(*self.vgg16_features[:-1]).to(self.device)
        #Using all the layers except the last Max pooling layer in the network
        # To subsample input image by a factor of 16 (<- subsample_ratio)
        
        
        for layer in range(10):
            for param in self.bbNet[layer].parameters():
                param.requires_grad = False
                
        #Classifier
        #Using all the classifier layers except the last layer
        self.classifier = list(self.vgg16.classifier)[:-1]
        self.classifier = nn.Sequential(*self.classifier).to(self.device)
        self.classifier_out_features = 4096
    
    def forward(self, img):
        """
        Args:
            img (Tensor or List(tensors)) :: given a list of images or a tensor of images, parse and return the feature maps
        """
        if isinstance(img, list):
            img = [i.to(self.device) for i in img]
        else:
            img = img.to(self.device)
        return self.bbNet(img)
    
    def getModel(self):
        """
        Get backbone network
        """
        return self.bbNet
    
    def getClassifier(self):
        """
        Get the classifier and classifier out features
        """
        return self.classifier, self.classifier_out_features

    def getOutChannels(self):
        #second last layer as the last layer is a ReLU layer
        return self.bbNet[-2].out_channels