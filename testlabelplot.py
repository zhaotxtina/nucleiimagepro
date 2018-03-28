# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:30:52 2018

@author: zt
"""

#%matplotlib inline
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation
import pandas as pd
import os
import random
from scipy import ndimage


image_ids = os.listdir("./input/stage1_test")
#%%
# Read prediction
def pred_test(pre,id, height, width):
    #pre="sub-dsbowl2018-test1.csv"  #,id, height, width
    sub=pd.read_csv(pre)

    sel=sub[sub.ImageId==id]
    num_pred=sel.shape[0]
    y_pred = np.zeros((num_pred, height, width), np.uint16)
    num_cnt=-1
    
    for i in sel.EncodedPixels:
        it = iter(i.split())
        num_cnt+=1
        for j in it:

            loc_h=int(j)%height
            loc_w=int(j)//height
            pred_l=int(next(it))
            #print j,loc_h,loc_w,pred_l
            y_pred[num_cnt,loc_h:(loc_h+pred_l),loc_w]=1
    #print("Number of true objects:", num_masks)
    print("Number of predicted objects:", num_pred)
    
    sumy_pred = np.zeros((height, width), np.uint16)
    sumy_pred[:,:] = np.sum(y_pred, axis=0)  # Add up to plot all masks
    
    return sumy_pred
#%%
def plot_test(y_pred): #,labels): 
    #y_pred=pre_test
    fig, axarr = plt.subplots(nrows=2,ncols=len(y_pred), figsize=(16,16))
    for j in range(len(y_pred)):
        ax = axarr[0,j]
        ax.imshow(y_pred[j])
        ax.set_title("Prediction")
    
    #labels[labels>0]=1

#    for j in range(len(y_pred)):
#        y_pred[j][y_pred[j]>0]=1
#        ax = axarr[1,j]
#        ax.imshow(y_pred[j]+labels)
#        ax.set_title("Overlapping")
#%%

id = random.sample(image_ids,1)[0]

# Load a single image and its associated masks
#id = '550450e4bff4036fd671decdc5d42fec23578198d6a2fd79179c4368b9d6da18'
file = "./input/stage1_test/{}/images/{}.png".format(id,id)
image = skimage.io.imread(file)
height, width, _ = image.shape

#print(image.shape)
#print(height, width,id)    
#show image and mask 
fig, axarr = plt.subplots(nrows=1,ncols=1, figsize=(16,16))
ax = axarr
ax.imshow(image)
ax.set_title("Original image")
#ax = axarr[1]
#ax.imshow(labels)
#ax.set_title("Ground truth masks")
images = np.zeros((height, width), np.uint16)
images[:,:] = np.sum(image, axis=2)  # Add up to plot all masks

pre_l1=pred_test("sub-dsbowl2018-test1.csv",id, height, width)
pre_l2=pred_test("submissiongraythres.csv",id, height, width)
#pre_l1=pred_test("sub-dsbowl2018.csv",id, height, width)
##pre_l1=pre_test("sub-dsbowl2018.csv",id, height, width)
##pre_l1=pre_train("tr_pre-bf_l1.csv",id, height, width, labels)
##mpre(labels,pre_l1)
##pre_dice=pre_train("tr_pre-dsbowl_dice.csv",id, height, width, labels)
#pre_dice=pred_test("sub-w_bin.csv",id, height, width)
#
##mpre(labels,pre_dice)
#pre_bincro=pred_test("sub-dvdw_bin.csv",id, height, width)

#
pre_test=[pre_l1,pre_l2] #,pre_dice,pre_bincro]
pre_test=[pre_l1] #,pre_dice,pre_bincro]

#plot_test(pre_test)
#plot_test(pre_l1,images)
y_pred=[images,pre_l1,pre_l2]
plot_test(y_pred)

    
