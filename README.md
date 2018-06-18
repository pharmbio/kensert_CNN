# BBBC021
# Generate images
1. Download all 55 zip files from https://data.broadinstitute.org/bbbc/BBBC021/.<br/>
2. Unzip and move all 55 folders to a folder in the main directory (where the scripts are).<br/>
3. Also download BBBC021_v1_image.csv and BBBC021_v1_moa.csv from same website and place in the same folder.
4. Run generate_dataset_bbbc021.py and enter the folder name containing all the image-folders.<br/>
5. Generated images will be saved to folder images_transformed_cropped/, and y labels will be outputted to a csv file.<br/>

# Train the models with the 38-fold cross-validation
1. Specify arguments of CNN_model in run_model.py if necessary.<br/>
2. Run run_model.py.<br/>
3. Prediction will be outputted to file.<br/>

# BBBC014
## Generate images
1. Download images from https://data.broadinstitute.org/bbbc/BBBC014/.<br/>
2. Move folder (BBBC014_v1_images) to same folder as the working directory.<br/>
3. Run generate_dataset_bbbc014.py.<br/>
4. Generated images will be saved to folder images_bbbc014/, and y labels will be outputted to a .npy file.<br/>

## Train the models with the 2-fold cross-validation
1. Specify arguments of CNN_model in run_model014.py if necessary.<br/>
2. Run run_model014.py.<br/>
3. Prediction will be outputted to file.<br/>
