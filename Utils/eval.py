import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import ExtendedEditDistance
"""
Perform COCO evaluation given the test loader and model
"""
def evaluate(model, test_loader, ious=None, device=torch.device('cpu'), custom=False, debug=False):
    model.eval()
    with torch.no_grad():
        #Get the evaluator
        if ious == None:
            map_evaluator = MeanAveragePrecision()
        else:
            map_evaluator = MeanAveragePrecision(iou_thresholds=ious)

        #Loop through test loader
        for images, targets in test_loader:
            images = [image.to(device) for image in images]
            targets = [{k:(v.to(device) if not(isinstance(v, list)) else v) for k,v in target.items()} for target in targets]

            if custom:
                #Case of custom implementation of model
                _, out = model(images)
            else:
                out = model(images)
                
            if debug:
                print(out[0]['boxes'].shape)
            
            #Update the evaluator with the output and the ground truth
            map_evaluator.update(out, targets)

        #Compute the metrics
        computed = map_evaluator.compute()
    
    return computed

def evaluate_text(gt_texts, preds):
    """
    :param gt_texts: List[List[Str]]
    :param preds: List[List[Str]]
    :return: Mean Extended Edit distance
    """

    score_total = 0
    nums = len(gt_texts)
    eed = ExtendedEditDistance()
    for gt_text, pred in zip(gt_texts, preds):
        score = eed(preds=pred, target=gt_text)

        score += float(score_total)

    return score_total / nums

