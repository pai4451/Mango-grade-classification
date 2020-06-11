import os, glob
from shutil import copyfile
from itertools import combinations

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# original data
train_data = './data/C1-P1_Train'
dev_data   = './data/C1-P1_Dev'

# processed data
train_proc = './processed/C1-P1_Train'
dev_proc = './processed/C1-P1_Dev'

for dir_ in [train_proc, dev_proc]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

for type_ in ['train', 'dev']:
    if not os.path.exists(f'./processed/{type_}.csv'):
        src = os.path.join('./data', f'{type_}.csv')
        dst = os.path.join('./processed', f'{type_}.csv')
        copyfile(src, dst)

# 2D Euclidean distances
Euclidean_dist = lambda x1, x2: np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)

def color_filter(hsv, img):
    global fail_num
    lower = {'red':(166, 84, 141), 'yellow':(21, 59, 119), 'orange':(0, 50, 80), 'green':(61, 122, 129)} 
    upper = {'red':(186,255,255), 'yellow':(60,255,255), 'orange':(20,255,255), 'green':(86,255,255)}

    # colors = {'red':(0,0,255), 'orange':(0,140,255), 'yellow':(0, 255, 217), 'green':(0,255,0)}

    best_x, best_y, best_ma, best_MA, best_angle = 0, 0, 0, 0, 0
    color_range = range(1, 3) # at most 2 color ranges
    for i in color_range:
        # Combine any combination of i colors
        for i_colors in combinations(['red', 'yellow', 'orange', 'green'], i):
            kernel = np.ones((9,9),np.uint8)
            masks = [cv2.inRange(hsv, lower[c], upper[c]) for c in i_colors]

            for id_, m in enumerate(masks):
                masks[id_] = cv2.morphologyEx(masks[id_], cv2.MORPH_OPEN, kernel)
                masks[id_] = cv2.morphologyEx(masks[id_], cv2.MORPH_CLOSE, kernel)

            mask = sum(masks)

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if len(cnts) != 0:
                for cont in cnts:
                    if len(cont) >= 5:
                        (x,y),(ma,MA),angle = cv2.fitEllipse(cont)
                        if ma > best_ma and MA > best_MA:
                            (best_x, best_y), (best_ma, best_MA), best_angle = (x,y),(ma,MA),angle

    ellipse = (best_x, best_y), (best_ma, best_MA), best_angle

    original_area = img.shape[0] * img.shape[1]
    ellipse_area = np.pi/4 * best_MA * best_ma
    area_ratio = ellipse_area/original_area

    original_mid  = (img.shape[1]/2, img.shape[0]/2)
    ellipse_mid = (best_x, best_y)
    dist_from_mid = Euclidean_dist(original_mid, ellipse_mid)
    dist_ratio = dist_from_mid/(0.5 * Euclidean_dist((0,0), img.shape))

    # area of ellipse enough and mid point of ellipse close mid of figure
    if area_ratio > 0.25 and dist_ratio <= 0.5:
        Ellipse = cv2.ellipse(np.zeros_like(img), ellipse, (255,255,255), -1)
        result = np.bitwise_and(img, Ellipse)
    else:
        result = img
        fail_num += 1

    return mask, result

fail_num = 0
def main():
    global fail_num
    for data in [train_data, dev_data]: # [train_data, dev_data]
        print(f"Processing {data} ...")
        data_img = glob.glob(os.path.join(data, "*.jpg"))
        print(len(data_img))
        for data_file in tqdm(data_img):
            # Load file
            img = cv2.imread(data_file)
            # Convert BGR to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask, result = color_filter(hsv, img)

            # Stack results
            # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # result = np.vstack((mask, result))

            # Save processed file
            proc_file = data_file.replace("data", "processed")
            cv2.imwrite(proc_file, result)
        print("Fail rate:", fail_num/len(data_img))
        fail_num = 0

if __name__ == '__main__':
    main()
