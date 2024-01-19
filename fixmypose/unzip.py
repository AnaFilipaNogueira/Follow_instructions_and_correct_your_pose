from zipfile import ZipFile 
#  
## loading the temp.zip and creating a zip object 
#with ZipFile("/nas-ctm01/datasets/public/fixmypose_reduced_imgs.zip", 'r') as zObject: 
#  
#    # Extracting all the members of the zip  
#    # into a specific location. 
#    zObject.extractall( 
#        path="/nas-ctm01/datasets/public/fixmypose2/") 


import os
import tarfile
dir_name = '/nas-ctm01/datasets/public/posefix_dataset/images'
extension = ".zip"

os.chdir(dir_name) # change directory from working dir to dir with files

for item in os.listdir(dir_name): # loop through items in dir
    if item.endswith(extension) and extension==".zip": # check for ".zip" extension
        file_name = os.path.abspath(item) # get full path of files
        zip_ref = ZipFile(file_name) # create zipfile object
        zip_ref.extractall(dir_name) # extract file to dir
        zip_ref.close() # close file
        #os.remove(file_name) # delete zipped file
    elif item.endswith(extension) and extension==".tar.bz2":
        file_name_tar = os.path.abspath(item) # get full path of files
        #zip_ref = ZipFile(file_name_tar)
        tar = tarfile.open(file_name_tar, "r:bz2")  
        tar.extractall()
    elif item.endswith(extension) and extension==".tar.xz":
        file_name_tar = os.path.abspath(item) # get full path of files
        #zip_ref = ZipFile(file_name_tar)
        tar = tarfile.open(file_name_tar)  
        tar.extractall()