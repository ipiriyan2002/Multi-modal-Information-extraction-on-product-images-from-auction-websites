import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision




def evaluate(model, test_loader, ious=None, device=torch.device('cpu'), custom=False):
    
    if not(custom):
        model.eval()
        model = model.to(device)
    else:
        device = model.getDevice()
    
    if ious == None:
        map_evaluator = MeanAveragePrecision()
    else:
        map_evaluator = MeanAveragePrecision(iou_thresholds=ious)
    
    for images, targets in test_loader:
        images = [image.to(device) for image in images]
        targets = [{k:v.to(device) for k,v in target.items()} for target in targets]
        
        if custom:
            out = model.inference(images)
        else:
            out = model(images)
        
        map_evaluator.update(out, targets)
    
    computed = map_evaluator.compute()
    
    return computed
        