# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
%matplotlib inline
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation
import pandas as pd
import os
import random
#%%
#sub=pd.read_csv("tr_pre-dsbowl_l1.csv")

# Read prediction
def pre_train(pre,id, height, width, labels):
    sub=pd.read_csv(pre)

    y_pred = np.zeros((height, width), np.uint16)
    num_pre=0
    sel=sub[sub.ImageId==id]
    for i in sel.EncodedPixels:
        it = iter(i.split())
        num_pre+=1
        for j in it:

            loc_h=int(j)%height
            loc_w=int(j)//height
            pred_l=int(next(it))
            #print j,loc_h,loc_w,pred_l
            y_pred[loc_h:(loc_h+pred_l),loc_w]=1+num_pre
    y_pred, _, _ = skimage.segmentation.relabel_sequential(y_pred) # Relabel objects
    # Compute number of objects
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    print("Number of true objects:", true_objects-1)
    print("Number of predicted objects:", pred_objects-1)

    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    #print iou #.diagonal()
    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        #print matches.shape
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # # Extra objects, predicted objects with no match
        false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

# Loop over IoU thresholds
    prec = []
    print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp*1.0 / (tp + fp + fn)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec))) 
    return y_pred
    
def plot_pre(y_pred,labels):    
    fig, axarr = plt.subplots(nrows=2,ncols=len(y_pred), figsize=(16,16))
    for j in range(len(y_pred)):
        ax = axarr[0,j]
        ax.imshow(y_pred[j])
        ax.set_title("Prediction")
    
    labels[labels>0]=1

    for j in range(len(y_pred)):
        y_pred[j][y_pred[j]>0]=1
        ax = axarr[1,j]
        ax.imshow(y_pred[j]+labels)
        ax.set_title("Overlapping")
        intersection = sum(sum(np.multiply(labels,y_pred[j])))
        print ("dice_coef", 2.0*intersection/(sum(sum(labels)) + sum(sum(y_pred[j]))) )   




#%%I don't have stage_train directory
#path="C:/Users/zwx462/Documents/kaggle/CNN/zexuan/input/stage1_test"
#image_ids = os.listdir(path)
image_ids = os.listdir("./input/stage1_train")
id = random.sample(image_ids,1)[0]

# Load a single image and its associated masks
#id = 'c395870ad9f5a3ae651b50efab9b20c3e6b9aea15d4c731eb34c0cf9e3800a72'
file = "./input/stage1_train/{}/images/{}.png".format(id,id)
masks = "./input/stage1_train/{}/masks/*.png".format(id)
image = skimage.io.imread(file)
masks = skimage.io.imread_collection(masks).concatenate()
height, width, _ = image.shape
num_masks = masks.shape[0]

# Make a ground truth label image (pixel value is index of object label)
labels = np.zeros((height, width), np.uint16)
for index in range(0, num_masks):
    labels[masks[index] > 0] = index + 1

pre_l1=pre_train("tr_pre-dsbowl_l1.csv",id, height, width, labels)
#mpre(labels,pre_l1)
pre_dice=pre_train("tr_pre-dsbowl_dice.csv",id, height, width, labels)
#mpre(labels,pre_dice)
pre_bincro=pre_train("tr_pre-dsbowl_bincro.csv",id, height, width, labels)

#show image and mask 
fig, axarr = plt.subplots(nrows=1,ncols=2, figsize=(16,16))
ax = axarr[0]
ax.imshow(image)
ax.set_title("Original image")
ax = axarr[1]
ax.imshow(labels)
ax.set_title("Ground truth masks")

#%%
pre=[pre_l1,pre_dice,pre_bincro]

plot_pre(pre,labels) 