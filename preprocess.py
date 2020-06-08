import os, glob
from shutil import copyfile
# from itertools import product

import cv2
import numpy as np
from matplotlib import pyplot as plt

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

def color_filter(hsv, img):
    lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 
        'yellow':(23, 59, 119), 'orange':(0, 50, 80)} 
    upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 
        'yellow':(54,255,255), 'orange':(20,255,255)}

    colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 
        'yellow':(0, 255, 217), 'orange':(0,140,255)}

    for key, value in upper.items():
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # centre = None

        if len(cnts) >= 5:
            c = max(cnts, key=cv2.contourArea)
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(img, ellipse, (0,0,0), 3)
            # M = cv2.moments(c)
            # centre = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # if radius > 0.5:
            #     cv2.ellipse(img, (int(x), int(y)), int(radius), colors[key], 2)

    return mask, img

def main():
    for data, proc in [(train_data, train_proc), (dev_data, dev_proc)]:
        break_counter = 0
        data_img = glob.glob(os.path.join(data, "*.jpg"))
        for data_file in data_img:
            # Load file
            img = cv2.imread(data_file)
            # Convert BGR to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask, img = color_filter(hsv, img)

            # Stack results
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            result = np.vstack((mask, img))

            # Save processed file
            proc_file = data_file.replace("data", "processed")
            cv2.imwrite(proc_file, result)
            if break_counter == 10:
                break
            break_counter += 1
        break

if __name__ == '__main__':
    main()
