# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:47:30 2018

@author: Setup01
"""

import pathlib2
import glob
import imageio
import numpy as np
from pathlib2 import Path
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import ListedColormap

## Glob the training data and load a single image path
#training_paths = pathlib2.Path('../stage1_train').glob('*/images/*.png')
#training_paths=glob.glob('../stage1_train/*/images/*.png')
#training_sorted = sorted([x for x in training_paths])
#im_path = training_sorted[45]
#im = imageio.imread(str(im_path))
##%%
## Print the image dimensions
#print('Original image shape: {}'.format(im.shape))
#
## Coerce the image into grayscale format (if not already)
#from skimage.color import rgb2gray
#im_gray = rgb2gray(im)
#print('New image shape: {}'.format(im_gray.shape))
#
##%%
## Now, let's plot the data
#import matplotlib.pyplot as plt
#
#plt.figure(figsize=(10,4))
#
#plt.subplot(1,2,1)
#plt.imshow(im)
#plt.axis('off')
#plt.title('Original Image')
#
#plt.subplot(1,2,2)
#plt.imshow(im_gray, cmap='gray')
#plt.axis('off')
#plt.title('Grayscale Image')
#
#plt.tight_layout()
#plt.show()
#
##%%
#
#from skimage.filters import threshold_otsu
#thresh_val = threshold_otsu(im_gray)
#mask = np.where(im_gray > thresh_val, 1, 0)
#
## Make sure the larger portion of the mask is considered background
#if np.sum(mask==0) < np.sum(mask==1):
#    mask = np.where(mask, 0, 1)
#    
#    #%%
#    
#plt.figure(figsize=(10,4))
#
#plt.subplot(1,2,1)
#im_pixels = im_gray.flatten()
#plt.hist(im_pixels,bins=50)
#plt.vlines(thresh_val, 0, 100000, linestyle='--')
#plt.ylim([0,50000])
#plt.title('Grayscale Histogram')
#
#plt.subplot(1,2,2)
#mask_for_display = np.where(mask, mask, np.nan)
#plt.imshow(im_gray, cmap='gray')
#plt.imshow(mask_for_display, cmap='rainbow', alpha=0.5)
#plt.axis('off')
#plt.title('Image w/ Mask')
#
#plt.show()
#
##%%
#
#from scipy import ndimage
#labels, nlabels = ndimage.label(mask)
#
#label_arrays = []
#for label_num in range(1, nlabels+1):
#    label_mask = np.where(labels == label_num, 1, 0)
#    label_arrays.append(label_mask)
#
#print('There are {} separate components / objects detected.'.format(nlabels))
#
##%%
#
## Create a random colormap
#from matplotlib.colors import ListedColormap
#rand_cmap = ListedColormap(np.random.rand(256,3))
#
#labels_for_display = np.where(labels > 0, labels, np.nan)
#plt.imshow(im_gray, cmap='gray')
#plt.imshow(labels_for_display, cmap=rand_cmap)
#plt.axis('off')
#plt.title('Labeled Cells ({} Nuclei)'.format(nlabels))
#plt.show()
#
##%%
#
#for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
#    cell = im_gray[label_coords]
#    
#    # Check if the label size is too small
#    if np.product(cell.shape) < 10: 
#        print('Label {} is too small! Setting to 0.'.format(label_ind))
#        mask = np.where(labels==label_ind+1, 0, mask)
#
## Regenerate the labels
#labels, nlabels = ndimage.label(mask)
#print('There are now {} separate components / objects detected.'.format(nlabels))
#
##%%
#
#fig, axes = plt.subplots(1,8, figsize=(10,6))
#
#for ii, obj_indices in enumerate(ndimage.find_objects(labels)[50:58]):
#    cell = im_gray[obj_indices]
#    axes[ii].imshow(cell, cmap='gray')
#    axes[ii].axis('off')
#    axes[ii].set_title('Label #{}\nSize: {}'.format(ii+1, cell.shape))
#
#plt.tight_layout()
#plt.show()
#
##%%
## Get the object indices, and perform a binary opening procedure
#two_cell_indices = ndimage.find_objects(labels)[1]
#cell_mask = mask[two_cell_indices]
#cell_mask_opened = ndimage.binary_opening(cell_mask, iterations=8)
##%%
#fig, axes = plt.subplots(1,4, figsize=(12,4))
#
#axes[0].imshow(im_gray[two_cell_indices], cmap='gray')
#axes[0].set_title('Original object')
#axes[1].imshow(mask[two_cell_indices], cmap='gray')
#axes[1].set_title('Original mask')
#axes[2].imshow(cell_mask_opened, cmap='gray')
#axes[2].set_title('Opened mask')
#axes[3].imshow(im_gray[two_cell_indices]*cell_mask_opened, cmap='gray')
#axes[3].set_title('Opened object')
#
#
#for ax in axes:
#    ax.axis('off')
#plt.tight_layout()
#plt.show()
#
##%%
#
#def rle_encoding(x):
#    '''
#    x: numpy array of shape (height, width), 1 - mask, 0 - background
#    Returns run length as list
#    '''
#    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
#    run_lengths = []
#    prev = -2
#    for b in dots:
#        if (b>prev+1): run_lengths.extend((b+1, 0))
#        run_lengths[-1] += 1
#        prev = b
#    return " ".join([str(i) for i in run_lengths])
#
#print('RLE Encoding for the current mask is: {}'.format(rle_encoding(label_mask)))

#%%
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
    and dump it into a Pandas DataFrame.
    '''
    # Read in data and convert to grayscale
    #im_id = im_path.parts[-3]
    im_sel = im_path.split("\\")
    im_sel2=im_sel[len(im_sel)-1]
    im_id=im_sel2[:-4]
    im = imageio.imread(str(im_path))
    im_gray = rgb2gray(im)
    
    # Mask out background and extract connected objects
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)    
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    
    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': im_id, 'EncodedPixels': (" ".join(str(x) for x in rle))})
            im_df = im_df.append(s, ignore_index=True)
    
    return im_df


def analyze_list_of_images(im_path_list):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    all_df = pd.DataFrame()
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df = all_df.append(im_df, ignore_index=True)
    
    return all_df

#%%
    


testing=glob.glob('./input/stage1_test/*/images/*.png')
#testing=glob.glob('C://Users/zhaotx/kaggle/CNN/stage1_test/*/images/*.png')
#testing=glob.glob('E:/backupMarch2018/documents/zhaotx/kaggle/CNN/stage1_test/*/images/*.png')
#testing=glob.glob('E:/backup_anadarko/Cdrive/Documents/kaggle/CNN/working/input/stage1_test/*/images/*.png')

df = analyze_list_of_images(list(testing))
#all_df = pd.DataFrame()
#for im_path in testing:
#    im_path=testing[0]
#    im_df = analyze_image(im_path)
#    all_df = all_df.append(im_df, ignore_index=True)
#str1=[45104, 8, 45359, 11, 45614, 13, 45869, 14, 46124, 15, 46380, 15, 46636, 14, 46893, 12, 47150, 9, 47407, 6]
#str2=join(str1)
#subori['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
#print((" ".join(str(x) for x in str1)))
df.to_csv('submissiongraythres.csv', index=None)