
import os
import glob

every_nth_image = '100'
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
    imageB_path = files_domainB[idx]
    print(imageA_path)
    print(imageB_path)
