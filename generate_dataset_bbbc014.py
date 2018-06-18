import re
import glob, os
from PIL import Image
from skimage.transform import resize
import numpy as np
import cv2

path = "BBBC014_v1_images/"
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def slide_window(img, dims=(256, 256)):
    window_height, window_width = dims
    y, x = img.shape[:2]
    col = 0
    crop_images = np.zeros((16,256,256,3))
    index = 0
    for i in range(y//window_height):
        row = 0
        for j in range(x//window_width):
            crop_images[index] = img[row:row+window_height, col:col+window_width, :]
            row += 256
            index += 1
        col += 256
    return crop_images

def normalization(x, sample=None):
    channels = x.shape[3]
    # anscombe
    x = x.astype(np.float32)
    x[:,:,:,1:] = 2.0*np.sqrt(x[:,:,:,1:] + 3.0/8.0)

    # Subtracting mean DMSO and divide std DMSO from/by non-DMSO images pixel-wise
    #for i in range(channels-1):
        #x[:,:,:,i+1] -= x[:,:,:,i+1].mean()
        #x[:,:,:,i+1] /= x[:,:,:,i+1].std()

    for j in range(x.shape[0]):

        # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        OldRange = (x[j,:,:,1:].max() - x[j,:,:,1:].min())
        NewRange = (1 - 0)
        x[j,:,:,1:] = (((x[j,:,:,1:] - x[j,:,:,1:].min()) * NewRange) / OldRange) + 0

        # Map to 8-bit int
        x[j,:,:,1:] *= 255
        img = Image.fromarray(x[j].astype('uint8'))

        # save image to folder
        if sample == 1:
            img.save("images_bbbc014/bbbc014_%s.png" % j)
        elif sample == 2:
            img.save("images_bbbc014/bbbc014_%s.png" % str(j+512))
        else: pass
    return

print("Functions defined...")
# Plate include 96 samples with 2 field of views, a total of 192 instances.
Ch1 = np.zeros((96, 1024, 1360))
Ch2 = np.zeros((96, 1024, 1360))

for infile, index in zip(sorted(glob.glob(path +"*.Bmp"), key=numericalSort), range(192)):
    img = Image.open(infile)
    if index < 96:
        Ch1[index]    = np.array(img)
    else:
        Ch2[index-96] = np.array(img)
print("Images read...")

# Stack the two field of views + matrix of zeros to create RGB (with R being zeros)
# This is done because later a pre-trained ResNet50 and InceptionV2 and V3 are going to be used.
X = np.zeros((96,1024,1360,3))
for i in range(X.shape[0]):
    img_stacked = np.dstack((np.zeros((1024,1360)), Ch1[i], Ch2[i]))
    X[i] = img_stacked
X = X.astype('uint8')
print("Images stacked...")
# four highest concetrations and four lowest
labels = np.repeat(np.tile([1,1,1,1,0,0,0,0], 8), 16)
np.save("bbbc014_labels", labels)

sample_1 = np.zeros((512,256,256,3))
sample_2 = np.zeros((512,256,256,3))
m = 0
for i,j in enumerate([0,  1, 2, 3, 8, 9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,32,33,34,35,36,37,38,39,44,45,46,47,
                      48,49,50,51,56,57,58,59,60,61,62,63,68,69,70,71,72,73,74,75,80,81,82,83,84,85,86,87,92,93,94,95]):
    img = cv2.resize(X[j], (1024,1024), interpolation=cv2.INTER_AREA).astype('uint8')
    crop_images = slide_window(img)

    for k in crop_images:
        #img = resize(k, (224,224,3), mode='reflect',  preserve_range=True).astype('uint8')
        #img = Image.fromarray(img)
        #img.save("images_bbbc014/bbbc014_%s.png" % m)
        if j < 48:
            sample_1[m] = k
        else:
            sample_2[m-512] = k

        m += 1
print("Relevant images cropped (with 'slide_window' function)...")
# Normalize each cell line
normalization(sample_1, sample=1)
normalization(sample_2, sample=2)
print("Images normalized...Done.")
