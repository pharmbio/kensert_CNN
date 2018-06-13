import numpy as np
import pandas as pd
import os
from PIL import Image

while True:
    q1 = input("Crop images? y or n: ")
    if q1 != "y" and q1 != "n":
        print("Please answer 'y' or 'n'")
        continue
    else:
        break
crops = 4 if q1 == "y" else 1

dirname = input("Enter directory name of plate folders: ")
if not (any("BBBC021_v1_image.csv" in s for s in os.listdir(dirname)) and any("BBBC021_v1_moa.csv" in s for s in os.listdir(dirname))):
    raise ValueError("BBBC021_v1_image.csv and BBBC021_v1_moa.csv need to be in directory")

def slide_window(img, dims=(512, 640)):
    window_height, window_width = dims
    y, x = img.shape[:2]
    crop_images = np.zeros((4,512,640,3)) # dtype="uint8"

    index = 0
    col = 0
    for i in range(y//window_height):
        row = 0
        for j in range(x//window_width):
            crop_images[index] = img[row:row+window_height, col:col+window_width, :]
            row += window_height
            index += 1
        col += window_width
    return crop_images

def gamma_cor(img, gamma):
    return 255 * (img/255)**(gamma)

def anscombe(x):
    x = x.astype(np.float32)
    return (2.0*np.sqrt(x + 3.0/8.0))

def inverse_anscombe(z):
    z = z.astype(np.float32)
    #return (z/2.0)**2 - 3.0/8.0
    return (1.0/4.0 * np.power(z, 2) +
            1.0/4.0 * np.sqrt(3.0/2.0) * np.power(z, -1.0) -
            11.0/8.0 * np.power(z, -2.0) +
            5.0/8.0 * np.sqrt(3.0/2.0) * np.power(z, -3.0) - 1.0 / 8.0)

def DMSO_normalization(x,y,idx,crops,rm_imgs):
    '''
    Mean DMSO per plate per channel is subtracted from each [non-DMSO] image pixel-wise.
      input x is the transformed non-DMSO image
      input y is the transformed DMSO image
    the output is the non-DMSO image that has been normalized by DMSO statistics
    '''
    channels = x.shape[3]
    x = x.astype(np.float32)

    # Subtracting mean DMSO and divide std DMSO from/by non-DMSO images pixel-wise
    for i in range(channels):
        x[:,:,:,i] -= y[:,:,:,i].mean()
        x[:,:,:,i] /= y[:,:,:,i].std()

    # Map values to 8-bit integers and save to file
    # if crops = 4, slide_window function is used to generate cropped images.
    for i,j in zip(range(x.shape[0]), idx):

        # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin

        # Map to 8-bit int
        if crops == 1:
            OldRange = (x[i,:,:,:].max() - x[i,:,:,:].min())
            NewRange = (255 - 0)
            xt = (((x[i,:,:,:] - x[i,:,:,:].min()) * NewRange) / OldRange) + 0
            #xt = gamma_cor(xt, gamma=0.5)
            img = Image.fromarray(xt.astype('uint8'))
            # Save images to directory
            img.save("images_transformed_full/BBBC021_MCF7_%s.png" % str(j))
        else:
            OldRange = (x[i,:,:,:].max() - x[i,:,:,:].min())
            NewRange = (255 - 0)
            xt = (((x[i,:,:,:] - x[i,:,:,:].min()) * NewRange) / OldRange) + 0
            imgs = slide_window(xt)
            for i,img in enumerate(imgs):
                if ((img > (NewRange/5.1)).sum()/img.size) <= 0.002:
                    rm_imgs.append(i+j)
                    print(rm_imgs)
                    continue
                img_cropped = Image.fromarray(img.astype("uint8"))
                # Save images to directory
                img_cropped.save("images_transformed_cropped/BBBC021_MCF7_%s.png" % str(j+i-len(rm_imgs)))

    return

# read mechanism file
moa  = pd.read_csv(dirname+'/BBBC021_v1_moa.csv')
# read data file which link images to compound/concentration.
# which can then be linked to moa file
data = pd.read_csv(dirname+'/BBBC021_v1_image.csv')

labels = []
# keep track of images
count = 0
# keep track of dataset
dataset = 1
# List specifying images to be removed
rm_imgs = []
# Iterate through different directories (different plates)
for f in (f for f in os.listdir(dirname) if 'Week' in f):
    # Assign new variable for current plate
    plate_data = data[data['Image_PathName_DAPI'].str.contains(f)]

    idx = []
    plate_X = []
    plate_Y = []
    # Iterate through current plate
    for index, row in plate_data.iterrows():
        # Exclude all taxol compounds except certain examples from Week 1 plates
        if row['Image_Metadata_Compound'] == "taxol":
            if not (
                    'Week1_' in row['Image_PathName_DAPI']  and
                    'D0' in row['Image_Metadata_Well_DAPI'] and
                        (
                        "0.3" in str(row['Image_Metadata_Concentration']) or
                        "1.0" in str(row['Image_Metadata_Concentration']) or
                        "3.0" in str(row['Image_Metadata_Concentration'])
                        )
                    ):
                continue

        # Extract compounds that have a MOA annotation
        if moa[(moa['compound']      == row['Image_Metadata_Compound']) &
               (moa['concentration'] == row['Image_Metadata_Concentration'])].shape[0] > 0:

            #Read the images
            img_DAPI    = Image.open(dirname+'/%s/%s' % (f, row['Image_FileName_DAPI']))
            img_DAPI    = np.array(img_DAPI)

            img_Tubulin = Image.open(dirname+'/%s/%s' % (f, row['Image_FileName_Tubulin']))
            img_Tubulin = np.array(img_Tubulin)

            img_Actin   = Image.open(dirname+'/%s/%s' % (f, row['Image_FileName_Actin']))
            img_Actin   = np.array(img_Actin)

            # Make it RGB (stack the three channels) and append to list of images of current plate
            img_stack   = np.dstack((img_Actin, img_Tubulin, img_DAPI))
            plate_X.append(img_stack)

            # Obtain mechanism, compound and concentration for image
            mechanism  = moa[(moa['compound']      == row['Image_Metadata_Compound']) &
                             (moa['concentration'] == row['Image_Metadata_Concentration'])]

            # Append additional labels (apart from mechanism, compounds, concentrations) to labels list.
            # And all different rotations/mirrors (x 8).
            if row['Image_Metadata_Compound'] != 'DMSO':
                [labels.append([mechanism.values.tolist()[0][0],
                               mechanism.values.tolist()[0][1],
                               mechanism.values.tolist()[0][2],
                               row['Image_Metadata_Plate_DAPI'],
                               row['Image_Metadata_Well_DAPI'],
                               row['Replicate']]) for i in range(crops)]

                idx.append(count)
                count += crops

            plate_Y.append([mechanism.values.tolist()[0][0],
                           mechanism.values.tolist()[0][1],
                           mechanism.values.tolist()[0][2],
                           row['Image_Metadata_Plate_DAPI'],
                           row['Image_Metadata_Well_DAPI'],
                           row['Replicate']])

    plate_Y = np.asarray(plate_Y)
    dmso_idx     = np.where(plate_Y[:,0] == "DMSO")[0]
    non_dmso_idx = np.where(plate_Y[:,0] != "DMSO")[0]

    if len(non_dmso_idx) > 0:
        plate_X = np.asarray(plate_X)
        plate_X = anscombe(plate_X)
        #plate_X = inverse_anscombe(plate_X)
        DMSO_normalization(plate_X[non_dmso_idx], plate_X[dmso_idx], idx, crops, rm_imgs)
    print('Number of compounds transformed = ' + str(count) + '; dataset = ' + str(dataset))
    dataset += 1


if crops == 1:
    df = pd.DataFrame(labels)
    df.to_csv('BBBC021_MCF7_labels_full.csv',
              header=["compound", "concentration", "moa", "plate", "well", "replicate"], sep=';')
else:
    for index in sorted(rm_imgs, reverse=True):
        del labels[index]
    df = pd.DataFrame(labels)
    df.to_csv('BBBC021_MCF7_labels_cropped.csv',
              header=["compound", "concentration", "moa", "plate", "well", "replicate"], sep=';')
