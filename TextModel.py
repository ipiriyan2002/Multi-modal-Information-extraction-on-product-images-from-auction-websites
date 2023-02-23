#Installing libraries
print("INSTALLING LIBRARIES....")
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
print("LIBRARIES INSTALLED!")


#Importing preprocess library defined by Author: Ishaipiriyan Karunakularatnam
import preprocess_dataset as pre


MAX_LABELS = pre.getMaxLabels()
IMG_SIZE = pre.getImgSize()


#CORE CONSTANTS
IMG_INDEX = 1 #Min is 0, Max is 799
BATCH = 16
EPOCHS = 50
L_RATE = 0.001
pos_alpha = 1.2
neg_alpha = 0.2


def combined_loss(y_true, y_pred):
    #Splitting the true and prediction into confidence scores and bounding boxes
    true_conf  = y_true[0]
    true_box  = y_true[1]
    pred_conf = y_pred[0]
    pred_box = y_pred[1]
   
    
    #Defining the weighting factors depending on the class
    pos_mask = backend.cast(backend.equal(true_conf, 1), backend.floatx()) * pos_alpha
    neg_maks = backend.cast(backend.equal(true_conf, 0), backend.floatx()) * neg_alpha
    
    #Calculating the confidence loss seperately for each confidence after applying mask
    #Then adding them together
    bce = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    conf_loss_pos = bce(true_conf * pos_mask, pred_conf * pos_mask)
    
    conf_loss_neg = bce(true_conf * neg_mask, pred_conf * neg_mask)
    
    conf_loss = conf_loss_pos + conf_loss_neg
    
    conf_loss = backend.mean(conf_loss)
    
    #Calculating the bounding box loss for all bounding boxes and then applying mask
    #Same reason as above but for bounding boxes
    #Manually defining mean squared error as to apply the mask
    true_box = backend.cast(true_box, 'float32')
    bbox_loss = backend.square(true_box - pred_box)
    bbox_loss = backend.sum(bbox_loss, axis=-1)
    bbox_loss = backend.mean(bbox_loss * pos_mask + bbox_loss * neg_mask)
    
    return conf_loss, bbox_loss


def averageGradients(conf_gradients, bbox_gradients):
    gradients = []
    for index, conf_grad in enumerate(conf_gradients[:28]):
        bbox_grad = bbox_gradients[index]
        
        if (conf_grad != None) and (bbox_grad != None):
            avg_grad = conf_grad + bbox_grad
            avg_grad /= 2

            gradients.append(avg_grad)
        elif (conf_grad == None):
            gradients.append(bbox_grad)
        else:
            gradients.append(conf_grad)
    
    
    return gradients


class TextDetectorModel(keras.Model):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.comb_loss_tracker = keras.metrics.Mean(name="comb_loss")
        self.conf_loss_tracker = keras.metrics.Mean(name="conf_loss")
        self.bbox_loss_tracker = keras.metrics.Mean(name="bbox_loss")
        self.conf_acc_tracker = keras.metrics.Accuracy(name="conf_accuracy")
        self.bbox_acc_tracker = keras.metrics.Accuracy(name="bbox_accuracy")
    
    def train_step(self, data):
        x, y = data
        
        conf_true, bbox_true = y[0], y[1]
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True) #Forward pass
            
            conf_pred, bbox_pred = y_pred[0], y_pred[1]
            
            conf_loss, bbox_loss = combined_loss(y, y_pred)
        
        #Compute the gradients
        trainable_vars = self.trainable_variables
        #print(len(trainable_vars))
        gradients_conf_loss = tape.gradient(conf_loss, trainable_vars)
        gradients_bbox_loss = tape.gradient(bbox_loss, trainable_vars)
        

        self.optimizer.apply_gradients(zip(gradients_conf_loss, trainable_vars))
        self.optimizer.apply_gradients(zip(gradients_bbox_loss, trainable_vars))

        #Compute the loss metrics
        self.conf_loss_tracker.update_state(conf_loss)
        self.bbox_loss_tracker.update_state(bbox_loss)
        self.comb_loss_tracker.update_state(conf_loss + bbox_loss)
        
        #Compute the accuracy metrics
        self.conf_acc_tracker.update_state(conf_true, conf_pred)
        self.bbox_acc_tracker.update_state(bbox_true, bbox_pred)
        return {"Combined Loss": self.comb_loss_tracker.result(), "Confidence Score Loss": self.conf_loss_tracker.result(),
               "Bounding Box Loss": self.bbox_loss_tracker.result(), "Confidence Score Accuracy": self.conf_acc_tracker.result(),
               "Bounding Box Accuracy": self.bbox_acc_tracker.result()}
    
    @property
    def metrics(self):
        return [self.comb_loss_tracker, self.conf_loss_tracker, self.bbox_loss_tracker, self.conf_acc_tracker, self.bbox_acc_tracker]


# In[154]:


def getFeatureExtracter():
    # Defining the input layer
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Using vgg16 as the feature extractor
    featureExtractor = VGG16(weights="imagenet", include_top=False, input_tensor=input_layer)
    
    return input_layer, featureExtractor


# In[155]:


def getModel():
    num_labels = MAX_LABELS
    
    input_layer, fe_layer = getFeatureExtracter()
    
    x = fe_layer.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    
    confScores = Dense(1 * num_labels, activation="sigmoid", name="Confidence_Scores")(x)
    Bbox_coords = Dense(4 * num_labels, activation="sigmoid", name="BBox_regression")(x)
    Bbox_coords_RS = Reshape((num_labels, 4), name="Bbox")(Bbox_coords)
    
    model = TextDetectorModel(inputs=input_layer, outputs=[confScores, Bbox_coords_RS])
    
    model.compile(optimizer="adam")
    return model
