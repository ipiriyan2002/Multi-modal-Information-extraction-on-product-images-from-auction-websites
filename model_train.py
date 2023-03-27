#Installing libraries
print("INSTALLING LIBRARIES....")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import os
print("LIBRARIES INSTALLED!")

#Importing preprocess library defined by Author: Ishaipiriyan Karunakularatnam
import preprocess_cord as pre
import TextModel as tm

# # Pre-Processing the dataset

# In[3]:


MAX_LABELS = pre.getMaxLabels()
IMG_WIDTH, IMG_HEIGHT = pre.getImgSize()


# In[4]:


src_path = "./cord-v2/data/"
tImages, tLabels = pre.getTrainDataset(src_path)


# In[5]:


print("Pre-processing the datasets...")

train_images, (train_bboxes, train_conf_scores) = pre.pre_process_images(tImages), pre.pre_process_labels(tLabels, dense=True)

print("Training Data: Processed")


#CORE CONSTANTS
BATCH = tm.getBatch()
EPOCHS = tm.getEpochs()


# In[8]:


training_dataset = tf.data.Dataset.from_tensor_slices((train_images,(train_conf_scores, train_bboxes)))
training_dataset = training_dataset.shuffle(buffer_size=len(train_images))
training_dataset = training_dataset.batch(BATCH)
training_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



def train(save_path, dataset, epoch_range):
    model = tm.getModel()
    
    callback_cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path, verbose=1, save_weights_only=True,
        save_freq= "epoch")
    
    train_history = model.fit(dataset, batch_size=BATCH, callbacks = [callback_cp], epochs=epoch_range)
    
    return model, train_history





if __name__ == "__main__":
    save_path = "./Saved Model/Model_BASE_50"
    os.mkdirs(save_path)
    
    save_path = save_path + "/"
    
    model, history = train(save_path, training_dataset, EPOCHS)

    model_save_path = save_path + "model"
    os.mkdir(model_save_path)
    model.save(model_save_path + "/")


