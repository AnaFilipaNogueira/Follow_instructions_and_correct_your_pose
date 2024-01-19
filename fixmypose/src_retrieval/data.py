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
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
#import nltk
#nltk.download('averaged_perceptron_tagger')
DATA_ROOT = "/nas-ctm01/datasets/public"

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_saver(img, path):

    with open(path, 'wb') as f:
        img.save(f)

DEBUG_FAST_NUMBER = 1000

class DiffDataset:
    def __init__(self, ds_name='fixmypose', split='train'):
        self.ds_name = ds_name
        self.split = split
        self.data = json.load(
            open(os.path.join(DATA_ROOT, self.ds_name, self.split + "_ret_NC.json"))
        )

        self.tok = Tokenizer()
        self.tok.load(os.path.join(DATA_ROOT, self.ds_name, "vocab_ret_NC.txt"))



class TorchDataset(Dataset):
    def __init__(self, dataset, max_length=80, 
                 img0_transform=None, img1_transform=None):
        self.dataset = dataset
        self.name = dataset.ds_name + "_" + dataset.split
        self.tok = dataset.tok
        self.max_length = max_length
        self.img0_trans, self.img1_trans = img0_transform, img1_transform

        f = h5py.File(os.path.join(DATA_ROOT, self.dataset.ds_name, 
            self.dataset.split + "_ret_NC_pixels.hdf5"), 'r')
        if args.fast:
            self.img0_pixels = f['img0'][:DEBUG_FAST_NUMBER]
            self.img1_pixels = f['img1'][:DEBUG_FAST_NUMBER]
        else:
            self.img0_pixels = f['img0']
            self.trg0_pixels = f['trg0']
            self.trg1_pixels = f['trg1']
            self.trg2_pixels = f['trg2']
            self.trg3_pixels = f['trg3']
            self.trg4_pixels = f['trg4']
            self.trg5_pixels = f['trg5']
            self.trg6_pixels = f['trg6']
            self.trg7_pixels = f['trg7']
            self.trg8_pixels = f['trg8']
            self.trg9_pixels = f['trg9']
            assert len(self.img0_pixels) == len(self.dataset.data), "%d, %d" % (len(self.img0_pixels),
                                                                                len(self.dataset.data))
            assert len(self.trg0_pixels) == len(self.dataset.data)
        


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
        ans_id = datum['ans_idx']
        
        # Load Image
        img_id = self.id2imgid[item]   

        img0 = torch.from_numpy(self.img0_pixels[img_id])      
        trg0 = torch.from_numpy(self.trg0_pixels[img_id])
        trg1 = torch.from_numpy(self.trg1_pixels[img_id])
        trg2 = torch.from_numpy(self.trg2_pixels[img_id])
        trg3 = torch.from_numpy(self.trg3_pixels[img_id])
        trg4 = torch.from_numpy(self.trg4_pixels[img_id])
        trg5 = torch.from_numpy(self.trg5_pixels[img_id])
        trg6 = torch.from_numpy(self.trg6_pixels[img_id])
        trg7 = torch.from_numpy(self.trg7_pixels[img_id])
        trg8 = torch.from_numpy(self.trg8_pixels[img_id])
        trg9 = torch.from_numpy(self.trg9_pixels[img_id])

        #################################### CUTOUT #########################################################################
        
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

        if args.alteracoes=='cutout_img0' or args.alteracoes=='cutout_img0_imgtrg' or args.alteracoes=='cutout_img0_img1':
            img_exp0 = self.img0_pixels[img_id].transpose((1, 2, 0))
            img0 = transforms_cutout(image=img_exp0)["image"]
        
        if args.alteracoes=='cutout_imgtrg' or args.alteracoes=='cutout_img0_imgtrg' or args.alteracoes=='cutout_img0_img1' or  args.alteracoes=='cutout_img1':
            trg0_exp = self.trg0_pixels[img_id].transpose((1, 2, 0))
            trg1_exp = self.trg1_pixels[img_id].transpose((1, 2, 0))
            trg2_exp = self.trg2_pixels[img_id].transpose((1, 2, 0))
            trg3_exp = self.trg3_pixels[img_id].transpose((1, 2, 0))
            trg4_exp = self.trg4_pixels[img_id].transpose((1, 2, 0))
            trg5_exp = self.trg5_pixels[img_id].transpose((1, 2, 0))
            trg6_exp = self.trg6_pixels[img_id].transpose((1, 2, 0))
            trg7_exp = self.trg7_pixels[img_id].transpose((1, 2, 0))
            trg8_exp = self.trg8_pixels[img_id].transpose((1, 2, 0))
            trg9_exp = self.trg9_pixels[img_id].transpose((1, 2, 0))
            trg0 = transforms_cutout(image=trg0_exp)["image"]
            trg1 = transforms_cutout(image=trg1_exp)["image"]
            trg2 = transforms_cutout(image=trg2_exp)["image"]
            trg3 = transforms_cutout(image=trg3_exp)["image"]
            trg4 = transforms_cutout(image=trg4_exp)["image"]
            trg5 = transforms_cutout(image=trg5_exp)["image"]
            trg6 = transforms_cutout(image=trg6_exp)["image"]
            trg7 = transforms_cutout(image=trg7_exp)["image"]
            trg8 = transforms_cutout(image=trg8_exp)["image"]
            trg9 = transforms_cutout(image=trg9_exp)["image"]
        ############################################################################################################################
        
        img0ID = datum['img0'].split("/")[-1].split(".")[0]
        trg0ID = datum['trg0'].split("/")[-1].split(".")[0]
        trg1ID = datum['trg1'].split("/")[-1].split(".")[0]
        trg2ID = datum['trg2'].split("/")[-1].split(".")[0]
        trg3ID = datum['trg3'].split("/")[-1].split(".")[0]
        trg4ID = datum['trg4'].split("/")[-1].split(".")[0]
        trg5ID = datum['trg5'].split("/")[-1].split(".")[0]
        trg6ID = datum['trg6'].split("/")[-1].split(".")[0]
        trg7ID = datum['trg7'].split("/")[-1].split(".")[0]
        trg8ID = datum['trg8'].split("/")[-1].split(".")[0]
        trg9ID = datum['trg9'].split("/")[-1].split(".")[0]


        if False or self.dataset.split == "train":
            sent = datum['sent']
            stop_words = ["waist", "hip", "hips", "leg","legs", "thigh","thighs","knee","knees","foot","feet","heels", "heel","toe","toes", "arm","arms",
            "forearms", "forearm","shoulder","shoulders", "hand","hands", "wrist", "wrists","palm", "palms", "finger", "fingers", "elbow", "elbows",
            "head", "face", "eyes", "forehead", "torso", "navel", "chest", "body", "belly","neck", "throat", "right", "left"]
            
            #print('-----------------------------------------------')
            #print(sent)
            # -------------------------------- Spelling Augmentation ------------------------------------------
            if args.nlpaug_choice == 'spelling_aug' or args.nlpaug_choice == 'all_nlpaug':
                if args.choose_stop_words == "True":
                    aug_spelling = naw.SpellingAug(aug_p=args.aug_p, aug_min=1, stopwords=stop_words)
                else:
                    aug_spelling = naw.SpellingAug(aug_p=args.aug_p, aug_min=1)
                #aug_spelling = naw.SpellingAug(aug_p=args.aug_p, aug_min=1, stopwords=args.choose_stop_words)
                sent = aug_spelling.augment(sent)
                #print('Spelling Aug')
                #print(sent)

            # -------------------------------- Delete Randomly ------------------------------------------------
            if args.nlpaug_choice == 'delete_random' or args.nlpaug_choice == 'all_nlpaug':
                if args.choose_stop_words == "True":
                    aug_random_delete = naw.RandomWordAug(aug_p=args.aug_p, aug_min=1, stopwords=stop_words)
                else:
                    aug_random_delete = naw.RandomWordAug(aug_p=args.aug_p, aug_min=1)
                #aug_random_delete = naw.RandomWordAug(aug_p=args.aug_p, aug_min=1, stopwords=args.choose_stop_words)
                sent = aug_random_delete.augment(sent)
                #print('Delete Aug')
                #print(sent)

            # -------------------------------- Synonym Replacement --------------------------------------------
            if args.nlpaug_choice == 'synonym_replace' or args.nlpaug_choice == 'all_nlpaug':
                if args.choose_stop_words == "True":
                    aug_synonym = naw.SynonymAug(aug_src='wordnet', aug_p=args.aug_p, aug_min=1, stopwords=stop_words)
                else:
                    aug_synonym = naw.SynonymAug(aug_src='wordnet', aug_p=args.aug_p, aug_min=1)
                #aug_synonym = naw.SynonymAug(aug_src='wordnet', aug_p=args.aug_p, aug_min=1, stopwords=args.choose_stop_words)
                sent = aug_synonym.augment(sent)
                #print('Synonym Aug')
                #print(sent)
            
            # -------------------------------- Sequential --------------------------------------------
            if args.nlpaug_choice == 'sequential':
                if args.choose_stop_words == "True":
                    aug_sequential = naf.Sequential([naw.RandomWordAug(aug_p=args.aug_p, aug_min=1, stopwords=stop_words), 
                    naw.SynonymAug(aug_src='wordnet', aug_p=args.aug_p, aug_min=1, stopwords=stop_words), 
                    naw.SpellingAug(aug_p=args.aug_p, aug_min=1, stopwords=stop_words)])
                else:
                    aug_sequential = naf.Sequential([naw.RandomWordAug(aug_p=args.aug_p, aug_min=1), 
                    naw.SynonymAug(aug_src='wordnet', aug_p=args.aug_p, aug_min=1), 
                    naw.SpellingAug(aug_p=args.aug_p, aug_min=1)])
                    
                sent = aug_sequential.augment(sent)
                #print('Sequential Aug')
                #print(sent)
            
            # -------------------------------- Sometimes --------------------------------------------
            if args.nlpaug_choice == 'sometimes':
                if args.choose_stop_words == "True":
                    aug_sometimes = naf.Sometimes([naw.RandomWordAug(aug_p=args.aug_p, aug_min=1, stopwords=stop_words), 
                    naw.SynonymAug(aug_src='wordnet', aug_p=args.aug_p, aug_min=1, stopwords=stop_words), 
                    naw.SpellingAug(aug_p=args.aug_p, aug_min=1, stopwords=stop_words)])
                else:
                    aug_sometimes = naf.Sometimes([naw.RandomWordAug(aug_p=args.aug_p, aug_min=1), 
                    naw.SynonymAug(aug_src='wordnet', aug_p=args.aug_p, aug_min=1), 
                    naw.SpellingAug(aug_p=args.aug_p, aug_min=1)])

                sent = aug_sometimes.augment(sent)
                #print('Sometimes Aug')
                #print(sent)
            
                

        else:
            sent = datum['sents'][0]
            
        inst = self.tok.encode(sent)
        length = len(inst)
        a = np.ones((self.max_length), np.int64) * self.tok.pad_id

        if length < self.max_length:        
            a[: length] = inst

            length = length
        else:                                        
            a[:] = inst[:self.max_length]

            length = self.max_length

        # Lang: numpy --> torch
        inst = torch.from_numpy(a)
        leng = torch.tensor(length)
        trg = (trg0,trg1,trg2,trg3,trg4,trg5,trg6,trg7,trg8,trg9)

        return uid, img0, trg, inst, leng, ans_id
        