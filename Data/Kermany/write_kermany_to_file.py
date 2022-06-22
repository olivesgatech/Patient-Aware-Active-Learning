from PIL import Image
import numpy as np
import os
import cv2 as cv

img_path = '../../../../../../../OCT/BIGandDATA/ZhangData/OCT/'
file_path = './'

train_files = open(os.path.join(file_path, 'train.txt'), "r")
test_files = open(os.path.join(file_path, 'test.txt'), "r")

train_list = train_files.readlines()
test_list = test_files.readlines()

train_list = [id_.rstrip().split(',') for id_ in train_list]
test_list = [id_.rstrip().split(',') for id_ in test_list]

raw_tr = np.array([[np.array(Image.open(os.path.join(img_path, 'train', fname[0].split('-')[0], fname[0]))),
                    int(fname[1]), int(fname[0].split('-')[1])]  # image, label, patient ID
                   for fname in train_list])
raw_te = np.array([[np.array(Image.open(os.path.join(img_path, 'test', fname[0].split('-')[0], fname[0]))),
                    int(fname[1]), int(fname[0].split('-')[1])]  # image, label, patient ID
                   for fname in test_list])

for i, img in enumerate(raw_tr[:, 0]):
    raw_tr[i, 0] = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA)

for i, img in enumerate(raw_te[:, 0]):
    raw_te[i, 0] = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA)

mean = np.mean(np.mean(raw_tr[:, 0] / 255.0))
print(mean)
std = np.std(np.std(raw_tr[:, 0] / 255.0))
print(std)

with open(os.path.join(file_path, 'train.npy'), 'wb') as f:
    np.save(f, raw_tr)

with open(os.path.join(file_path, 'test.npy'), 'wb') as f:
    np.save(f, raw_te)

