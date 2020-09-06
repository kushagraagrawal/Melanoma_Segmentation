from __future__ import print_function

import os
import gzip
import numpy as np

import cv2

from constants import *

def preprocessor(input_img):
    output_img = np.ndarray((input_img.shape[0], input_img.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(input_img.shape[0]):
        output_img[i, 0] = cv2.resize(input_img[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return output_img

def create_train_data(data_path, masks_path):
    """
    Generate training data numpy arrays and save them into the project path
    """

    image_rows = 420
    image_cols = 580

    images = os.listdir(data_path)
    masks = os.listdir(masks_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    print(imgs.shape)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    i = 0

    for image_name in images:
        if image_name.endswith("jpg"):
            img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE)
        else:
            continue
        try:
            img = cv2.resize(img, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
            img = np.array([img])
            imgs[i] = img
            i+=1
        except Exception as e:
            print("file name:", image_name)
            print("index:", i)
            print(e)
        # print(img.shape)
    print("images done, masks now....")
    i=0

    for image_mask_name in masks:
        if image_name.endswith("jpg"):
            img_mask = cv2.imread(os.path.join(masks_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        else:
            continue
        img_mask = cv2.resize(img_mask, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
        img_mask = np.array([img_mask])
        imgs_mask[i] = img_mask
        i+=1

    print("Successfully done!")

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)


def load_train_data():
    """
    Load training data from project path
    :return: [X_train, y_train] numpy arrays containing the training data and their respective masks.
    """
    print("\nLoading train data...\n")
    X_train = np.load('imgs_train.npy')
    y_train = np.load('imgs_mask_train.npy')

    X_train = preprocessor(X_train)
    y_train = preprocessor(y_train)

    X_train = X_train.astype('float32')

    mean = np.mean(X_train)  # mean for data centering
    std = np.std(X_train)  # std for data normalization

    X_train -= mean
    X_train /= std

    y_train = y_train.astype('float32')
    y_train /= 255.  # scale masks to [0, 1]
    return X_train, y_train


if __name__ == '__main__':
    create_train_data('isic-challenge-2017/ISIC-2017_Training_Data','isic-challenge-2017/ISIC-2017_Training_Part1_GroundTruth')
