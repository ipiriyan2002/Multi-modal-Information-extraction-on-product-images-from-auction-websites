import torch
import utils
import preprocess as pre
import data_augmentation as aug

class TextDetectorDataset(torch.utils.data.Dataset):
    def __init__(self, images, bboxes, conf_scores):
        self.images = images
        self.bboxes = bboxes
        self.conf_scores = conf_scores
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        bbox = self.bboxes[idx]
        conf_score = self.conf_scores[idx]
        
        # Convert the image and labels to PyTorch tensors
        img = torch.tensor(img)
        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        conf_score = torch.as_tensor(conf_score, dtype=torch.int64)
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        
        ground_truth = {}
        ground_truth["bboxes"] = bbox
        ground_truth["confidence scores"] = conf_score
        ground_truth["area"] = area
        
        return img, ground_truth

def getCordTorchDatasetLoader(config_file, split):
    #Load the config file
    config = utils.load_config_file(config_file)
    #Load the images and ground truths for cord dataset
    imgs, gts = utils.load_cord(split=split)
    #Preprocess the images and ground truths
    pImgs= pre.preprocess_images(imgs, config['IMG_WIDTH'], config['IMG_HEIGHT'])
    pBoxes, pConfs = pre.preprocess_cord_prices(gts, config['MAX_LABELS'])
    #Apply Transformations on the images and bboxes
    tImgs, tBoxes, tConfs = aug.getTransformedDataset(pImgs, pBoxes, pConfs, 
                                                      config['IMG_WIDTH'], config['IMG_HEIGHT'], config['MAX_LABELS'])
    
    #Create a custom dataset from the numpy arrays
    dataset = TextDetectorDataset(tImgs, tBoxes, tConfs)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=config['BATCH'], shuffle=True)
    
    return dataset_loader

