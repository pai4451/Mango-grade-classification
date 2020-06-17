import os, glob
from shutil import copyfile
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# original data
train_data = './data/C1-P1_Train'
dev_data   = './data/C1-P1_Dev'

# image by folder
train_proc = './data/training'
dev_proc = './data/validation'
class_ = ['A','B','C']
for dir_ in [train_proc, dev_proc]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    for i in class_:
        if not os.path.exists(dir_ + '/' + i):
            os.makedirs(dir_ + '/' + i) 


df = pd.read_csv('./data/train.csv')
for i in range(len(df)):
    if not os.path.exists('./data/training' + '/' +  df['label'][i] + '/' + df['image_id'][i] ):
        src = os.path.join('./data/C1-P1_Train' ,df['image_id'][i])
        dst = os.path.join('./data/training', df['label'][i], df['image_id'][i])
        copyfile(src, dst)

df = pd.read_csv('./data/dev.csv')
for i in range(len(df)):
    if not os.path.exists('./data/validation' + '/' +  df['label'][i] + '/' + df['image_id'][i] ):
        src = os.path.join('./data/C1-P1_Dev' ,df['image_id'][i])
        dst = os.path.join('./data/validation', df['label'][i], df['image_id'][i])
        copyfile(src, dst)