#import pickle

#pkl_file = pickle.load(open("/nas-ctm01/datasets/public/posefix_dataset/posefix_release/vocab_posefix_6157_pp4284_auto.pkl", "rb"))
#list_words = list(pkl_file['word2idx'])

#with open('/nas-ctm01/datasets/public/posefix_dataset/vocab.txt', 'w') as f:
#    for i in list_words:
#        f.write(i)
#        f.write("\n")

#import h5py
#f1 = h5py.File('/nas-ctm01/datasets/public/fixmypose/train_NC_pixels.hdf5', 'r')
#print(list(f1.keys()))
#print(len(f1))

#import json

#split = "train_in_sequence"
#f = open('/nas-ctm01/datasets/public/posefix_dataset/' + split + '.json')

#file_read = json.load(f)
#list_of_img0 = []
#list_of_img1 = []

#for i in range(len(file_read)):
#    list_of_img0.append(file_read[i]['img0'])
#    list_of_img1.append(file_read[i]['img1'])

#f.close()

#print(len(list_of_img0), len(list_of_img1))
#print(list_of_img0, list_of_img1)


import json

split = "train_in_sequence"
f = open('/nas-ctm01/datasets/public/posefix_dataset/' + split + '.json')

file_read = json.load(f)
#list_of_img0 = []
#list_of_img1 = []

for i in range(len(file_read)):
    list_of_img0.append(file_read[i]['img0'])
    list_of_img1.append(file_read[i]['img1'])

#f.close()

#print(len(list_of_img0), len(list_of_img1))
#print(list_of_img0, list_of_img1)
