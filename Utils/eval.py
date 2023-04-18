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
            
            if custom:
                images = images.to(device)
                
                targets_final =[]
                for index, _ in enumerate(targets['boxes']):
                    dict_ = {k:v[index].to(device) for k,v in targets.items()}
                    targets_final.append(dict_)
                
                out_dict = model.inference(images)
                
                out = []
                for index, box in enumerate(out_dict['boxes']):
                    dict_ = {
                        "boxes":box,
                        "scores":out_dict['scores'][index],
                        "labels":out_dict['labels'][index]
                    }
                    
                    out.append(dict_)
                
                map_evaluator.update(out, targets_final)
                
            else:
                images = [image.to(device) for image in images]
                targets = [{k:v.to(device) for k,v in target.items()} for target in targets]
                
                out = model(images)

                map_evaluator.update(out, targets)

        computed = map_evaluator.compute()
    
    return computed
        