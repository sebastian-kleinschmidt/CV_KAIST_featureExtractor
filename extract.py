import os
import glob
import numpy as np
import math
import cv2

##
# Helper
##


def distance(pointA, pointB):
    dist = math.sqrt((pointB[0]-pointA[0])**2+(pointB[1]-pointA[1])**2)
    return dist

def patchValid(corner, patch_size, img):
    img_size = img.shape
    corner[0] = int(corner[0])
    corner[1] = int(corner[1])
    lr_dist = (patch_size-1)/2

    if corner[0]-lr_dist>0 and corner[1]-lr_dist>0 and corner[0]+lr_dist<img_size[0] and corner[1]+lr_dist<img_size[1]:
        return True
    return False

def extractPatch(img, corner, patch_size):
    corner[0] = int(corner[0])
    corner[1] = int(corner[1])
    lr_dist = (patch_size-1)/2
    lr_dist = int(lr_dist)
    patch = img[corner[0]-lr_dist:corner[0]+lr_dist,corner[1]-lr_dist:corner[1]+lr_dist]
    return patch

##
# Variables
##

every_nth_image = 100    #Only take every n-th image
image_size = [512,640]
allow_feat_dist = 5.0        #Alloewd distance between intermodal features
feat_max = 250
feat_quality_level = 0.01
feat_dist = 25.0
patch_size = 301

root_dir = '/home/deeplearning/Code/Python/DL_DomainIndependentImageTransfer/images/'

domain_A_folder = 'lwir/'
domain_B_folder = 'visible/'
image_A_type = '.jpg'
image_B_type = '.jpg'

current_working_dir = ''
output_dir = os.getcwd()
output_filetype = '.jpg'
suffix_A = 'thermal_keypoint'
suffix_B = 'rgb_keypoint'

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
for img_idx, imageA_path in enumerate(files_domainA):
    #split_path
    path_split = imageA_path.split('/')
    c_set = path_split[-4]
    c_version = path_split[-3]
    #Check if folder exist, otherwise create
    if not os.path.isdir(output_dir+'/'+c_set):
        os.mkdir(output_dir+'/'+c_set)

    if not os.path.isdir(output_dir+'/'+c_set+'/'+c_version):
        os.mkdir(output_dir+'/'+c_set+'/'+c_version)

    if math.fmod(img_idx, every_nth_image)==0:
        if not os.path.isdir(output_dir+'/'+c_set+'/'+c_version+'/'+str(img_idx)):
            os.mkdir(output_dir+'/'+c_set+'/'+c_version+'/'+str(img_idx))

        imageB_path = files_domainB[img_idx]
        #Load Images
        imgA = cv2.imread(imageA_path)
        imgA_gray = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
        imgB = cv2.imread(imageB_path)
        imgB_gray = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)

        #Detect features in both images
        cornersA = cv2.goodFeaturesToTrack(imgA_gray,feat_max, feat_quality_level, feat_dist)
        #cornersA_plot = np.int0(cornersA)
        cornersB = cv2.goodFeaturesToTrack(imgB_gray,feat_max, feat_quality_level, feat_dist)

        #Filter Corners
        cornersA_accepted = []
        cornersB_accepted = []

        for idxA, cornerA in enumerate(cornersA):
            min_idxA = -1
            min_idxB = -1
            min_dist = 99999.0;
            for idxB, cornerB in enumerate(cornersB):
                temp_dist = distance(cornerA[0],cornerB[0])
                if temp_dist<min_dist:
                    min_dist = temp_dist
                    min_idxA = idxA
                    min_idxB = idxB
            if min_dist<allow_feat_dist:
                cornersA_accepted.append(cornersA[min_idxA][0])
                cornersB_accepted.append(cornersB[min_idxB][0])

        cornersA_accepted_plot = np.int0(cornersA_accepted)
        cornersB_accepted_plot = np.int0(cornersB_accepted)

        #Extract patches
        for idx in range(len(cornersA_accepted_plot)):
            if patchValid(cornersA_accepted_plot[idx], patch_size, imgA_gray) and patchValid(cornersB_accepted_plot[idx], patch_size, imgB_gray):
                print('Processing: '+c_set+'/'+c_version+'/'+str(img_idx)+'/kp_' + str(idx))
                patchA = extractPatch(imgA, cornersA_accepted_plot[idx], patch_size)
                patchB = extractPatch(imgB, cornersB_accepted_plot[idx], patch_size)  
                #Generate file pathes
                if not os.path.isdir(output_dir+'/'+c_set+'/'+c_version+'/'+str(img_idx)+'/kp_' + str(idx)):
                    os.mkdir(output_dir+'/'+c_set+'/'+c_version+'/'+str(img_idx)+'/kp_' + str(idx))
                pathA = output_dir+'/'+c_set+'/'+c_version+'/'+str(img_idx)+'/kp_' + str(idx) + '/' + suffix_A + output_filetype
                pathB = output_dir+'/'+c_set+'/'+c_version+'/'+str(img_idx)+'/kp_' + str(idx) + '/' + suffix_B + output_filetype
                #Write patches
                cv2.imwrite(pathA, patchA);
                cv2.imwrite(pathB, patchB);

                #Visualize Output
                #cv2.imshow('imageA',patchA)
                #cv2.imshow('imageB',patchB)
                #cv2.waitKey(10)
                #cv2.destroyAllWindows()
