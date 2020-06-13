import os, glob
from shutil import copyfile
from itertools import combinations

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# # original data
# train_data = './data/C1-P1_Train'
# dev_data   = './data/C1-P1_Dev'

# # processed data
# train_proc = './processed/C1-P1_Train'
# dev_proc = './processed/C1-P1_Dev'


C1_P1_dir = glob.glob(os.path.join('./data', "C1-P1_*"))
for dir_ in C1_P1_dir:
    proc_dir = dir_.replace("data", "processed")
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)

data_csv = glob.glob(os.path.join('./data', "*.csv"))
for csv_file in data_csv:
    proc_csv = csv_file.replace("data", "processed")
    if not os.path.exists(proc_csv):
        copyfile(csv_file, proc_csv)

def color_filter(hsv, img):
    global fail_num

    # limit = 2.0
    # grid = (8,8)
    # clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)

    # cl = clahe.apply(l)
    # limg = cv2.merge((cl,a,b))

    # img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # lower = {'red':(166, 84, 141), 'yellow':(21, 59, 119), 'orange':(0, 50, 80), 'green':(61, 122, 129)} 
    # upper = {'red':(186,255,255), 'yellow':(60,255,255), 'orange':(20,255,255), 'green':(86,255,255)}

    # colors = {'red':(0,0,255), 'orange':(0,140,255), 'yellow':(0, 255, 217), 'green':(0,255,0)}

    # half_diagonal = np.linalg.norm(np.zeros(2)-img.shape)

    # best_x, best_y, best_ma, best_MA, best_angle = 0, 0, 0, 0, 0
    # color_range = range(1, 3) # at most 2 color ranges
    # for i in color_range:
    #     # Combine any combination of i colors
    #     for i_colors in combinations(['red', 'yellow', 'orange', 'green'], i):
    #         kernel = np.ones((9,9),np.uint8)
    #         masks = [cv2.inRange(hsv, lower[c], upper[c]) for c in i_colors]

    #         for id_, m in enumerate(masks):
    #             masks[id_] = cv2.morphologyEx(masks[id_], cv2.MORPH_OPEN, kernel)
    #             masks[id_] = cv2.morphologyEx(masks[id_], cv2.MORPH_CLOSE, kernel)

    #         mask = sum(masks)

    #         cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    #         if len(cnts) != 0:
    #             for cont in cnts:
    #                 if len(cont) >= 5:
    #                     (x,y),(ma,MA),angle = cv2.fitEllipse(cont)
    #                     dist_from_mid = np.linalg.norm(original_mid-np.array((x, y)))
    #                     dist_ratio = dist_from_mid/(0.5 * half_diagonal)
    #                     if ma > best_ma and MA > best_MA and (angle < 30 or angle > 60) and dist_ratio <= 0.5:
    #                         (best_x, best_y), (best_ma, best_MA), best_angle = (x,y),(ma,MA),angle

    # ellipse = (best_x, best_y), (best_ma, best_MA), best_angle

    # original_area = img.shape[0] * img.shape[1]
    # ellipse_area = np.pi/4 * best_MA * best_ma
    # area_ratio = ellipse_area/original_area

    # # area of ellipse enough and mid point of ellipse close mid of figure
    # if area_ratio > 0.3:
    #     Ellipse = cv2.ellipse(np.zeros_like(img), ellipse, (255,255,255), -1)
    #     result = np.bitwise_and(img, Ellipse)

    original_mid  = np.array([img.shape[1]/2, img.shape[0]/2])
    mask = None

    ellipse = original_mid, 1.6*original_mid, 0
    Ellipse = cv2.ellipse(np.zeros_like(img), ellipse, (255,255,255), -1)
    result = np.bitwise_and(img, Ellipse)

    # 1: vertical; 0: horizontal
    vertical = 1 if img.shape[0] >= img.shape[1] else 0
    if vertical:
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

    return mask, result

fail_num = 0
def main():
    global fail_num
    for data in C1_P1_dir:
        print(f"Processing {data} ...")
        data_img = glob.glob(os.path.join(data, "*.jpg"))
        print("Number of data:", len(data_img))
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
    # pass
