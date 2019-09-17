import os
import glob
import numpy as np
import math
import cv2

every_nth_image = 50    #Only take every n-th image
allow_feat_dist = 5        #Alloewd distance between intermodal features
feat_max = 1000
feat_quality_level = 0.01
feat_dist = 5

root_dir = '/home/deeplearning/Code/Python/DL_DomainIndependentImageTransfer/images/'

domain_A_folder = 'lwir/'
domain_B_folder = 'visible/'
image_A_type = '.jpg'
image_B_type = '.jpg'

##
# Load Data
##

#Find all sets of the dataset
files_sets = sorted(glob.glob(root_dir+'*', recursive=True))
#Find all version for each set
file_versions = []
for file_set in files_sets:
    temp = sorted(glob.glob(file_set+'/*', recursive=True))
    file_versions.append(temp)
#List all images in folder
files_domainA = []
files_domainB = []

for folder in file_versions:
    for subfolder in folder:
        tempA = sorted(glob.glob(subfolder+'/'+domain_A_folder+"*"+image_A_type, recursive=True))
        tempB = sorted(glob.glob(subfolder+'/'+domain_B_folder+"*"+image_B_type, recursive=True))
        if len(tempA)>0 and len(tempB)>0:
            for imageA in tempA:
                files_domainA.append(imageA)
            for imageB in tempB:  
                files_domainB.append(imageB)

if(len(files_domainA)!=len(files_domainA) or len(files_domainA)==0):
	print('Critical Error - Unequal number of images')

##
# Process Images and Detect features
##
for idx, imageA_path in enumerate(files_domainA):
    if math.fmod(idx, every_nth_image)==0:
        imageB_path = files_domainB[idx]
        #Load Images
        imgA = cv2.imread(imageA_path)
        imgA = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
        imgB = cv2.imread(imageB_path)
        imgB = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)
        #Detect features in both images
        cornersA = cv2.goodFeaturesToTrack(imgA,feat_max, feat_quality_level, feat_dist)
        cornersA_plot = np.int0(cornersA)
        cornersB = cv2.goodFeaturesToTrack(imgB,feat_max, feat_quality_level, feat_dist)
        cornersB_plot = np.int0(cornersB)
        #Visualize Features
        for i in cornersA_plot:
            x,y = i.ravel()
            cv2.circle(imgA,(x,y),3,255,-1)
        for i in cornersB_plot:
            x,y = i.ravel()
            cv2.circle(imgB,(x,y),3,255,-1)

        #Visualize Images
        cv2.imshow('image',imgA)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('image',imgB)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
