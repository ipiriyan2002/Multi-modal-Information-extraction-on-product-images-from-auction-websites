from Model.one_stage_detector import OneStageDetector
import os
import numpy as np
import pandas as pd
import torch
import io
import utils
import preprocess as pre
import data_augmentation as aug
import DataLoader as dl

def train_model(model, trainLoader, epochs, device, lRate=1e-3):
    print("Starting Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    optimizer = torch.optim.Adam(model.parameters(), lr=lRate)
    
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.train()
    
    all_losses = []
    
    printer_stop = 10
    
    for i in range(epochs):
        tot_loss = 0
        for img_batch, target_batch in trainLoader:
            bbox_batch = target_batch['bboxes']
            conf_batch = target_batch['confidence scores']
            img_batch = img_batch.to(device)
            bbox_batch = bbox_batch.to(device)
            conf_batch = conf_batch.to(device)
            loss = model(img_batch, bbox_batch, conf_batch)
            
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            
            tot_loss += loss.mean().item()
        
        all_losses.append(tot_loss)
        if ((i + 1) % printer_stop == 0):
            print("Epoch {0} ->  Mean Loss :: {1}".format(i, sum(all_losses)/len(all_losses)))
    
    return loss_list

if __name__ == "__main__":
    config = utils.load_config_file("params_cord_initial.yaml")

    train_loader = dl.getCordTorchDatasetLoader("params_cord_initial.yaml", split='train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Found Device: ",device)
    model = OneStageDetector(config['IMG_WIDTH'], config['IMG_HEIGHT'], 512, 16)
    print("Model Defined")
    all_losses = train_model(model, train_loader, config['EPOCHS'], config['L_RATE'])
    print("Model Trained")
    torch.save(model.state_dict(), config['SAVE_PATH'])
