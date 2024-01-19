from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
from param import args
from tok import Tokenizer
from utils import BufferLoader
import copy
from PIL import Image
import json
import random
import os
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DATA_ROOT = "/nas-ctm01/datasets/public"

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_saver(img, path):
    print(img, path)
    with open(path, 'wb') as f:
        img.save(f)

DEBUG_FAST_NUMBER = 1000

class DiffDataset: #produzir os tokens dando o vocabulario
    def __init__(self, ds_name='fixmypose', split='train'):
        self.ds_name = ds_name
        self.split = split
        
        if args.dataset == "fixmypose":
            self.data = json.load(
                open(os.path.join(DATA_ROOT, self.ds_name, self.split + "_NC.json"))
            )

            self.tok = Tokenizer()
            self.tok.load(os.path.join(DATA_ROOT, self.ds_name, "vocab_NC.txt"))
        
        elif args.dataset == "posefix":
            self.data = json.load(
                open(os.path.join(DATA_ROOT, self.ds_name, self.split + ".json"))
            )#[:100]

            self.tok = Tokenizer()
            self.tok.load(os.path.join(DATA_ROOT, self.ds_name, "vocab.txt"))


class TorchDataset(Dataset):
    def __init__(self, dataset, max_length=80, 
                 img0_transform=None, img1_transform=None):
        self.dataset = dataset
        self.name = dataset.ds_name + "_" + dataset.split
        self.tok = dataset.tok
        self.max_length = max_length
        self.img0_trans, self.img1_trans = img0_transform, img1_transform

        if args.dataset == "fixmypose":
            f = h5py.File(os.path.join(DATA_ROOT, self.dataset.ds_name, 
                self.dataset.split + "_NC_pixels.hdf5"), 'r')
            if args.fast:
                self.img0_pixels = f['img0'][:DEBUG_FAST_NUMBER]
                self.img1_pixels = f['img1'][:DEBUG_FAST_NUMBER]
            else:
                self.img0_pixels = f['img0']
                self.img1_pixels = f['img1']
                assert len(self.img0_pixels) == len(self.dataset.data), "%d, %d" % (len(self.img0_pixels),
                                                                                    len(self.dataset.data))
                assert len(self.img1_pixels) == len(self.dataset.data)

        elif args.dataset == "posefix":
            f = open(os.path.join(DATA_ROOT, "posefix", self.dataset.split + ".json"))

            file_read = json.load(f)#[:100]
            img0_pixels_list = []
            img1_pixels_list = []

            for i in range(len(file_read)):
                #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                #img_transform = transforms.Compose([transforms.Resize((args.resize, args.resize)), transforms.ToTensor(), normalize])
                
                img0_to_append = Image.open(file_read[i]['img0'])
                img0_to_append = img0_to_append.resize((224, 224))
                #img0_to_append = img_transform(img0_to_append)
                img0_to_append = np.array(img0_to_append) #, dtype=float
                img0_to_append = np.reshape(img0_to_append, (np.shape(img0_to_append)[2], np.shape(img0_to_append)[1], np.shape(img0_to_append)[0]))
                img0_pixels_list.append(img0_to_append)
                
                img1_to_append = Image.open(file_read[i]['img1'])
                img1_to_append = img1_to_append.resize((224, 224))
                #img1_to_append = img_transform(img1_to_append)
                img1_to_append = np.array(img1_to_append) #, dtype=float
                img1_to_append = np.reshape(img1_to_append, (np.shape(img1_to_append)[2], np.shape(img1_to_append)[1], np.shape(img1_to_append)[0]))
                img1_pixels_list.append(img1_to_append)
            
            self.img0_pixels = img0_pixels_list
            self.img1_pixels = img1_pixels_list

            assert len(self.img0_pixels) == len(self.dataset.data), "%d, %d" % (len(self.img0_pixels), len(self.dataset.data))
            assert len(self.img1_pixels) == len(self.dataset.data)
        
        if False or self.dataset.split == "train":
            self.train_data = []
            self.id2imgid = {}
            for i, datum in enumerate(self.dataset.data):
                if args.fast and i >= DEBUG_FAST_NUMBER:     
                    break
                for sent in datum['sents']:
                    new_datum = datum.copy()
                    new_datum.pop('sents')
                    new_datum['sent'] = sent
                    self.id2imgid[len(self.train_data)] = i     
                    self.train_data.append(new_datum)
        
        elif False or self.dataset.split == "train_in_sequence" or self.dataset.split == "train_out_sequence":
            self.train_data = []
            self.id2imgid = {}
            for i, datum in enumerate(self.dataset.data):
                if args.fast and i >= DEBUG_FAST_NUMBER:     
                    break
                for sent in datum['sents_0']:
                    new_datum = datum.copy()
                    new_datum.pop('sents_0')
                    new_datum['sent'] = sent
                    self.id2imgid[len(self.train_data)] = i     
                    self.train_data.append(new_datum)

        else:
            self.train_data = []
            self.id2imgid = {}
            for i, datum in enumerate(self.dataset.data):
                if args.fast and i >= DEBUG_FAST_NUMBER:     
                    break

                self.id2imgid[len(self.train_data)] = i    
                self.train_data.append(datum)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        datum = self.train_data[item]
        uid = datum['uid']
        
        # Load Image
        img_id = self.id2imgid[item]        

        img0 = torch.from_numpy(self.img0_pixels[img_id])      
        img1 = torch.from_numpy(self.img1_pixels[img_id])

        #################################### CUTOUT #########################################################################
        img_exp0 = self.img0_pixels[img_id].transpose((1, 2, 0))
        img_exp1 = self.img1_pixels[img_id].transpose((1, 2, 0))
        
        transforms_cutout = A.Compose([ 
            A.CoarseDropout(
                max_holes = 20,         # Maximum number of regions to zero out. (default: 8)
                max_height = 60,        # Maximum height of the hole. (default: 8) 
                max_width = 60,         # Maximum width of the hole. (default: 8) 
                min_holes=1,            # Maximum number of regions to zero out. (default: None, which equals max_holes)
                min_height=1,           # Maximum height of the hole. (default: None, which equals max_height)
                min_width=1,            # Maximum width of the hole. (default: None, which equals max_width)
                fill_value=0,           # value for dropped pixels.
                mask_fill_value=None,   # fill value for dropped pixels in mask. 
                always_apply=False, 
                p=args.cutout_p),
                ToTensorV2(),])

        if args.alteracoes=='cutout_img0':
            img0 = transforms_cutout(image=img_exp0)["image"]
        elif args.alteracoes=='cutout_img1':
            img1 = transforms_cutout(image=img_exp1)["image"]
        elif args.alteracoes=='cutout_img0_img1':
            img0 = transforms_cutout(image=img_exp1)["image"]
            img1 = transforms_cutout(image=img_exp1)["image"]
        else:
            img0 = torch.from_numpy(self.img0_pixels[img_id])      
            img1 = torch.from_numpy(self.img1_pixels[img_id])
        ############################################################################################################################
        
        img0ID = datum['img0'].split("/")[-1].split(".")[0]
        img1ID = datum['img1'].split("/")[-1].split(".")[0]

        if False or self.dataset.split == "train":
            sent = datum['sent']
        elif False or self.dataset.split == "train_in_sequence" or self.dataset.split == "train_out_sequence":
            sent = datum['sent']
        else:
            sent = datum['sents_0'][0] #qdo usar FixMypose alterar para datum['sents'], se for o posefix datum['sents_0'][0]

        sent = sent.replace(".", "").replace(",", "") 
        inst = self.tok.encode(sent)
        length = len(inst)
        a = np.ones((self.max_length), np.int64) * self.tok.pad_id
        a[0] = self.tok.bos_id
        if length + 2 < self.max_length:        
            a[1: length+1] = inst
            a[length+1] = self.tok.eos_id
            length = 2 + length
        else:                                           
            a[1: -1] = inst[:self.max_length-2]
            a[self.max_length-1] = self.tok.eos_id      
            length = self.max_length

        inst = torch.from_numpy(a)
        leng = torch.tensor(length)

        return uid, img0, img1, inst, leng
        
