import torch
import torch.nn as nn
import torchvision

class BackboneNetwork(nn.Module):
    def __init__(self, device=None):
        super(BackboneNetwork, self).__init__()
        #self.cuda_device = "cuda" if device == None else device
        self.device = device#torch.device(self.cuda_device if torch.cuda.is_available() else "cpu")
        vgg16_model = torchvision.models.vgg16(pretrained=True)
        vgg16_model = vgg16_model.to(self.device)
        vgg16_features = list(vgg16_model.features)
        
        self.bbNet = nn.Sequential(*vgg16_features[:-1])
        self.bbNet = self.bbNet.to(self.device)
        #Using all the layers except the last Max pooling layer in the network
        # To subsample input image by a factor of 16 (<- subsample_ratio)
    
    def forward(self, img):
        img = img.to(self.device)
        return self.bbNet(img)
