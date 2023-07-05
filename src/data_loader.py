from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

from src.utils import rle_to_mask, rle_codes_to_mask


class DataGenerator(keras.utils.Sequence):
    def __init__(self, 
                 image_paths:List[str],
                 dataframe_labels:pd.DataFrame=None,
                 batch_size=32,
                 image_size=(768,768),
                 augmentations = None,
                 shuffle=False):
        """
        Initialize the DataGenerator object.
        
        Arguments:
        - image_paths: List of image file paths.
        - dataframe_labels: DataFrame containing image labels.
        - batch_size: Number of samples per batch.
        - image_size: Tuple specifying the target image size.
        - augmentations: Optional image augmentations (albumentations).
        - shuffle: Boolean indicating whether to shuffle the data after each epoch.
        """
        self.image_paths = image_paths
        self.dataframe_labels = dataframe_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.image_size = image_size
        self.on_epoch_end()
        
    def on_epoch_end(self):
        # reshuffle in the end
        if self.shuffle == True:
            self.image_paths = np.random.shuffle(self.image_paths)

    def __len__(self):
        return int(np.floor(len(self.image_paths)/self.batch_size)) 
    
    def __getitem__(self,index):
        start_index = index*self.batch_size
        end_index = (index+1)*self.batch_size
        image_batch_paths = self.image_paths[start_index:end_index]
        X,y = self.__generate_data(image_batch_paths)
        return X,y
        
    def __generate_data(self, image_batch_paths):
        """
        Generate the input images and masks for a batch of image paths.
        
        Arguments:
        - image_batch_paths: List of image file paths for the batch.
        
        Returns:
        - Tuple (X, y) containing the input images and corresponding masks for the batch.
        """
        # init batches
        X = np.empty((self.batch_size,*self.image_size,3),dtype=np.float32)
        y = np.empty((self.batch_size,*self.image_size,1),dtype=np.float32)
        for i, image_path in enumerate(image_batch_paths):
            # read image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            # read mask
            rle_codes = self.dataframe_labels[self.dataframe_labels["image_path"]==image_path]["EncodedPixels"].values
            mask = rle_codes_to_mask(rle_codes,image.shape[:2])
            # resize to model format
            mask = cv2.resize(mask.astype(np.uint8),(self.image_size[1],self.image_size[0]))
            image = cv2.resize(image,(self.image_size[1],self.image_size[0]))
            # use augmentations
            if self.augmentations:
                augmented = self.augmentations(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            # converting image and mask to appropriate format 
            image = image.astype(np.float32)
            image = image / 255.0
            mask = mask.astype(np.float32)
            mask = mask[...,None]
            X[i] = image
            y[i] = mask
            
        return X,y

