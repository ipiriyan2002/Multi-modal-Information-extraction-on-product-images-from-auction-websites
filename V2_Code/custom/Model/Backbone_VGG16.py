import torch
import torch.nn as nn
import torchvision

class BackboneNetwork(nn.Module):
    def __init__(self, device=None):
        super(BackboneNetwork, self).__init__()
        self.device = device if device != None else torch.device('cpu')
        vgg16_model = torchvision.models.vgg16(pretrained=True)
        vgg16_model = vgg16_model.to(self.device)
        vgg16_features = list(vgg16_model.features)
        
        self.bbNet = nn.Sequential(*vgg16_features[:-1]).to(self.device)
        #Using all the layers except the last Max pooling layer in the network
        # To subsample input image by a factor of 16 (<- subsample_ratio)
        
        
        for layer in range(10):
            for param in self.bbNet[layer].parameters():
                param.requires_grad = False
    
    def forward(self, img):
        img = img.to(self.device)
        return self.bbNet(img)
    
    def getModel(self):
        return self.bbNet
