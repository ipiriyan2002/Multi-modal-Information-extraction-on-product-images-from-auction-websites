import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision




def evaluate(model, test_loader, ious=None, device=torch.device('cpu'), custom=False):
    model.eval()
    with torch.no_grad():
        if ious == None:
            map_evaluator = MeanAveragePrecision()
        else:
            map_evaluator = MeanAveragePrecision(iou_thresholds=ious)

        for images, targets in test_loader:
            images = [image.to(device) for image in images]
            targets = [{k:v.to(device) for k,v in target.items()} for target in targets]
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            if custom:
                out = model.inference(images)
            else:
                out = model(images)

            map_evaluator.update(out, targets)

        computed = map_evaluator.compute()
    
    return computed
        