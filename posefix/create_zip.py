import shutil

output_filename = "/home/up201705948/posescript-main/posefix_images"
dir_name = "images_saved"
root_dir = "/home/up201705948/posescript-main/"

shutil.make_archive(output_filename, 'zip', root_dir, dir_name)



#import zipfile

#zp = zipfile.ZipFile("/home/up201705948/posescript-main/posefix_images.zip")

#size = sum([zinfo.file_size for zinfo in zp.filelist])
#zip_kb = float(size) / 1000  # kB
#print('The size is ', zip_kb*10**(-6), 'GB')


