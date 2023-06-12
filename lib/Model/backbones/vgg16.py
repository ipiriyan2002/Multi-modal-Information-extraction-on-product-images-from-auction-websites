import torch
import torch.nn as nn
import torchvision

class VGG16_BACKBONE():
    """
    VGG16 backbone network and also their classifier
    """
    def __init__(self, pretrained=False, pretrained_path="data/pretrained/vgg16/vgg16-397923af.pth"):
        
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.loaded = False
        self.vgg16 = torchvision.models.vgg16()
    
    def loadWeights(self):
        if self.pretrained and not(self.loaded):
            print("Loading pretrained weights...")
            print(f"Weights loaded from {self.pretrained_path}")
            state_dict = torch.load(self.pretrained_path)
            self.vgg16.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg16.state_dict()})
            self.loaded = True
            
        
    
    def getModel(self):
        """
        Get backbone network
        """
        self.loadWeights()
        bbNet = nn.Sequential(*list(self.vgg16.features)[:-1])
        for layer in range(10):
            for param in bbNet[layer].parameters():
                param.requires_grad = False
        
        return bbNet

    def getClassifier(self):
        
        self.loadWeights()

        classifier_list = list(self.vgg16.classifier)

        #Not using last layer
        del classifier_list[6]
        #And not using drop out layer
        del classifier_list[5]
        del classifier_list[2]

        classifier = nn.Sequential(*classifier_list)
        out_channels = 4096
        
        return classifier, out_channels
    
    def roiResize(self, pooled):
        return pooled.view(pooled.size(dim=0), -1)
    
    def getOutChannels(self):
        #second last layer as the last layer is a ReLU layer
        return 512

    def getFeatScaler(self):
        return 16