import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch import nn
import segmentation_models_pytorch as smp
import os
import albumentations as albu
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
from collections import OrderedDict
from catalyst import dl

from catalyst.core import Callback, CallbackOrder
from catalyst.dl.callbacks import AccuracyCallback, CheckpointCallback, AUCCallback, CriterionCallback, MetricAggregationCallback, MeterMetricsCallback, VerboseLogger, SchedulerCallback, OptimizerCallback
from catalyst.dl.callbacks.metrics.iou import IouCallback
from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint
from albumentations.pytorch.transforms import ToTensor
import base64
from tqdm import tqdm
from catalyst.contrib.nn.criterion.iou import BCEIoULoss
from scipy import ndimage, stats

FP16 = True
DATASET_PATH = "train"
TEST_DATASET_PATH = "test"
LABELS = ['staphylococcus_epidermidis', 'klebsiella_pneumoniae', 'staphylococcus_aureus', 'moraxella_catarrhalis', 'c_kefir', 'ent_cloacae']


class ITSDataset(Dataset):
    def __init__(self, filelist, train_transforms=None, blur_mask = False):
        self.blur_mask = blur_mask
        self.filelist = filelist
        transform_list = [albu.Normalize(), ToTensor()]
        if train_transforms:
            #train_transforms = [albu.VerticalFlip(), albu.HorizontalFlip(), albu.ShiftScaleRotate()]
            transform_list = train_transforms + transform_list
        self.augs = albu.Compose(transform_list)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        basename = self.filelist[idx]
        im = cv2.imread(os.path.join(DATASET_PATH, basename + ".png"))
        with open(os.path.join(DATASET_PATH, basename + ".json"), 'r') as f:
            layout = json.load(f)

        label = layout['shapes'][0]['label']
        label_ind = LABELS.index(label)

        h, w = layout['imageHeight'], layout['imageWidth']
        true_mask = np.zeros((h, w), np.uint8)
    

        for shape in layout['shapes']:
            polygon = np.array([point[::-1] for point in shape['points']])
            cv2.fillPoly(true_mask, [polygon[:, [1, 0]]], 255)
        if self.blur_mask:
            true_mask = cv2.GaussianBlur(true_mask,(3,3),0)
        transformed = self.augs(image = im, mask = true_mask)
        return {"image": transformed["image"], "mask": transformed["mask"], "label": label_ind}


class ITSDatasetWithPL(Dataset):
    def __init__(self, filelist, df_pl, train_transforms=None, blur_mask = False):
        self.blur_mask = blur_mask
        self.filelist = filelist
        self.df_pl = df_pl
        transform_list = [albu.Normalize(), ToTensor()]
        if train_transforms:
            #train_transforms = [albu.VerticalFlip(), albu.HorizontalFlip(), albu.ShiftScaleRotate()]
            transform_list = train_transforms + transform_list
        self.augs = albu.Compose(transform_list)

    def __len__(self):
        return len(self.filelist) + len(self.df_pl)

    def getitem_train(self, idx):
        basename = self.filelist[idx]
        im = cv2.imread(os.path.join(DATASET_PATH, basename + ".png"))
        with open(os.path.join(DATASET_PATH, basename + ".json"), 'r') as f:
            layout = json.load(f)

        label = layout['shapes'][0]['label']
        label_ind = LABELS.index(label)

        h, w = layout['imageHeight'], layout['imageWidth']
        true_mask = np.zeros((h, w), np.uint8)
    

        for shape in layout['shapes']:
            polygon = np.array([point[::-1] for point in shape['points']])
            cv2.fillPoly(true_mask, [polygon[:, [1, 0]]], 255)

        return im, true_mask
    
    def getitem_pl(self, idx):
        bn = self.df_pl.iloc[idx, 0]
        im = cv2.imread(os.path.join("test", str(bn).zfill(3) + ".png"))
        #print(os.path.join("test", str(bn).zfill(3) + ".png"))
        with open('tmp_bacteria.png', 'wb') as fp:
            fp.write(base64.b64decode(self.df_pl.iloc[idx, 2].encode()))
        mask = cv2.imread('tmp_bacteria.png', 0)
        #mask = cv2.GaussianBlur(mask,(3,3),0)
        #print(im, mask)
        return im, mask
    
    def __getitem__(self, idx):

        if idx < len(self.filelist):
            im, true_mask = self.getitem_train(idx)
        else:
            im, true_mask = self.getitem_pl(idx - len(self.filelist))
        if self.blur_mask:
            true_mask = cv2.GaussianBlur(true_mask,(3,3),0)
        transformed = self.augs(image = im, mask = true_mask)
        return {"image": transformed["image"], "mask": transformed["mask"]}

class ITSDataset4c(Dataset):
    def __init__(self, filelist, train_transforms=None, blur_mask = False):
        self.blur_mask = blur_mask
        self.filelist = filelist
        #self.df_sub = df_sub
        transform_list = [ToTensor()]
        self.normalize = albu.Normalize()
        if train_transforms:
            #train_transforms = [albu.VerticalFlip(), albu.HorizontalFlip(), albu.ShiftScaleRotate()]
            transform_list = train_transforms + transform_list
        self.augs = albu.Compose(transform_list)

    def __len__(self):
        return len(self.filelist)

    def getitem_train(self, idx):
        basename = self.filelist[idx]
        im = cv2.imread(os.path.join(DATASET_PATH, basename + ".png"))
        with open(os.path.join(DATASET_PATH, basename + ".json"), 'r') as f:
            layout = json.load(f)

        label = layout['shapes'][0]['label']
        label_ind = LABELS.index(label)

        h, w = layout['imageHeight'], layout['imageWidth']
        true_mask = np.zeros((h, w), np.uint8)
    

        for shape in layout['shapes']:
            polygon = np.array([point[::-1] for point in shape['points']])
            cv2.fillPoly(true_mask, [polygon[:, [1, 0]]], 255)

        return im, true_mask
    
    def __getitem__(self, idx):

        im, true_mask = self.getitem_train(idx)

        mask_aug = np.zeros_like(true_mask)
        true_mask_labeled = ndimage.label(true_mask)[0]
        for comp in ndimage.find_objects(true_mask_labeled):
            #use_comp = np.random.choice([True, False], p=[0.8, 0.2])
            #if not use_comp:
            #    continue
            dx = np.random.choice(range(6))
            dy = np.random.choice(range(6))
            target_slice = (slice(comp[0].start + dy, comp[0].stop + dy, None),
                            slice(comp[1].start + dx, comp[1].stop + dx, None))
            if true_mask[comp].shape == mask_aug[target_slice].shape:
                mask_aug[target_slice] = true_mask[comp]
        im = np.dstack([self.normalize(image=im)["image"], mask_aug / 255.])
       
        if self.blur_mask:
            true_mask = cv2.GaussianBlur(true_mask,(3,3),0)
        transformed = self.augs(image = im, mask = true_mask)
        return {"image": transformed["image"], "mask": transformed["mask"]}

class ITSTestDataset(Dataset):
    def __init__(self, filelist):
        self.filelist = filelist
        transform_list = [albu.Normalize(), ToTensor()]
        self.augs = albu.Compose(transform_list)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        basename = self.filelist[idx]
        im = cv2.imread(os.path.join(TEST_DATASET_PATH, basename + ".png"))
        
        transformed = self.augs(image = im)
        return {"image": transformed["image"], "id": basename}


class ITSTestDataset4C(Dataset):
    def __init__(self, filelist, test_mask_dir):
        self.filelist = filelist
        self.test_mask_dir = test_mask_dir#"decoded_ss_rotate"
        transform_list = [ToTensor()]
        self.normalize = albu.Normalize()
        self.augs = albu.Compose(transform_list)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        basename = self.filelist[idx]
        im = cv2.imread(os.path.join(TEST_DATASET_PATH, basename + ".png"))
        mask_aug = cv2.imread(os.path.join(self.test_mask_dir, basename + ".png"), cv2.IMREAD_GRAYSCALE)
        im = np.dstack([self.normalize(image=im)["image"], mask_aug / 255.])
        
        transformed = self.augs(image = im)
        return {"image": transformed["image"], "id": basename}

criterion = {"ce": nn.CrossEntropyLoss(),
            "bc": nn.BCEWithLogitsLoss(),
            "bciou": BCEIoULoss()}



device = dl.utils.get_device()
print(device)

fp16_params = None
if FP16:
    fp16_params=dict(opt_level="O1")

def encode_mask(m):
    cv2.imwrite("tmp_img.png", m.squeeze())
    with open("tmp_img.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
