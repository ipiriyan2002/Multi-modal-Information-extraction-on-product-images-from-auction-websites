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
                
            out_dict = model(images)
            
            if custom:
                out = []
                for index, box in enumerate(out_dict['boxes']):
                    dict_ = {
                        "boxes":box,
                        "scores":out_dict['scores'][index],
                        "labels":out_dict['labels'][index]
                    }
                    
                out.append(dict_)
            else:
                out = out_dict

            map_evaluator.update(out, targets)

        computed = map_evaluator.compute()
    
    return computed
"""
if custom:
                images = images.to(device)
                
                targets_final =[]
                for index, box in enumerate(targets['boxes']):
                    pos = torch.unique(torch.where(box >= 0)[0])
                    dict_ = {k:v[index][pos].to(device) for k,v in targets.items() if k != "image_id"}
                    dict_["image_id"] = targets["image_id"][index]
                    targets_final.append(dict_)
                
                out_dict = model.module.inference(images)
                
                out = []
                for index, box in enumerate(out_dict['boxes']):
                    dict_ = {
                        "boxes":box,
                        "scores":out_dict['scores'][index],
                        "labels":out_dict['labels'][index]
                    }
                    
                    out.append(dict_)
                
                map_evaluator.update(out, targets_final)
                
            else:"""