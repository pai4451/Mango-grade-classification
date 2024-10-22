import os, glob
from shutil import copyfile
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def png2jpg(folder_path, save_path, fill_color = (0,0,0)):
    '''folder_path: transparent png image generated by u2net 
       https://github.com/OPHoperHPO/image-background-remove-tool
       fill_color: (0,0,0) = black, (255, 255, 255) = white
    '''
    filename = os.listdir(folder_path)
    for file in filename:
        img = Image.open(folder_path + '/' + file)
        img = img.convert("RGBA")
        if img.mode == "RGBA":
            background = Image.new(img.mode[:-1], img.size, fill_color)
            background.paste(img, img.split()[-1]) # omit transparency
            img = background
        img.convert("RGB").save(save_path + '/' + os.path.splitext(file)[0] + '.jpg')



# train_transp, val_transp, test_transp are the folders contain image after remove background by u2net 
# For removing background by u2net, please check: https://github.com/OPHoperHPO/image-background-remove-tool
# e.g. `python3 main.py -i ./data/C1-P1_Train -o ./data/train_transp -m u2net``

train_proc = './data/train_transp'
val_proc = './data/val_transp'
test_proc = './data/test_transp'
trasparent_image = [train_proc, val_proc, test_proc]
# save path of jpg image
generated_image_black = ['./data/train_black', './data/val_black', './data/test_black']

# convert 4 channel png to 3 channel jpg
for i in range(len(generated_image_black)):
    if not os.path.exists(generated_image_black[i]):
        os.makedirs(generated_image_black[i])
    png2jpg(folder_path = trasparent_image[i], save_path = generated_image_black[i], fill_color = (0,0,0))


# original data
train_data = './data/train_black'
dev_data   = './data/val_black'

# image by folder
train_proc = './data/training2'
dev_proc = './data/validation2'

# generate folder by classes, later used by torchvision.datasets.ImageFolder
class_ = ['A','B','C']
for dir_ in [train_proc, dev_proc]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    for i in class_:
        if not os.path.exists(dir_ + '/' + i):
            os.makedirs(dir_ + '/' + i) 


df = pd.read_csv('./data/train.csv')
for i in range(len(df)):
    if not os.path.exists('./data/training2' + '/' +  df['label'][i] + '/' + df['image_id'][i] ):
        src = os.path.join('./data/train_black' ,df['image_id'][i])
        dst = os.path.join('./data/training2', df['label'][i], df['image_id'][i])
        copyfile(src, dst)

df = pd.read_csv('./data/dev.csv')
for i in range(len(df)):
    if not os.path.exists('./data/validation2' + '/' +  df['label'][i] + '/' + df['image_id'][i] ):
        src = os.path.join('./data/val_black' ,df['image_id'][i])
        dst = os.path.join('./data/validation2', df['label'][i], df['image_id'][i])
        copyfile(src, dst)