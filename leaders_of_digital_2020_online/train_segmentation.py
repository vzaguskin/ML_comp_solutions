from catalyst.dl import utils
SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

import its_training_utils as tu
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch import nn
import os
import json
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
import torchvision.models as models
from catalyst.contrib.nn.optimizers.radam import RAdam
from catalyst.contrib.nn.optimizers.lookahead import Lookahead
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
import segmentation_models_pytorch as smp
from catalyst.contrib.nn import FocalLossMultiClass, OneCycleLRWithWarmup
from catalyst.data.sampler import BalanceClassSampler
import gc
import time

import sys
RESUME = False
if  len(sys.argv) > 1 and sys.argv[1] in ["train", "resume"]:
    TRAINING = True
    if sys.argv[1] == "resume":
        RESUME = True
else:
    TRAINING = False

BATCH_SIZE = 4
ENCODER = 'efficientnet-b6'
RESUME_FOLD =0
THRESH = 0.5
WEIGHTS = 'imagenet'
ARCHITECTURE = smp.Unet


def getLogdir(training_mode):
    return './logs/catalyst_segmentation'

BASE_LOGDIR = getLogdir(TRAINING)
FP16 = True
DATASET_PATH = "train"
TEST_DATASET_PATH = "test"
LABELS = ['staphylococcus_epidermidis', 'klebsiella_pneumoniae', 'staphylococcus_aureus', 'moraxella_catarrhalis', 'c_kefir', 'ent_cloacae']

EPOCHS = 50

if __name__ == "__main__":

    image_files = [x.split(".")[0] for x in os.listdir("train") if ".png" in x]
    image_files = np.array(image_files)

    def getLabel(basename):
        with open(os.path.join(DATASET_PATH, basename + ".json"), 'r') as f:
            layout = json.load(f)

        label = layout['shapes'][0]['label']
        label_ind = LABELS.index(label)
        return label_ind

    image_labels = [getLabel(f) for f in image_files]

    #df_pl = pd.read_csv("submission_seg_leaky.csv")

    class DeepLabWrapper(nn.Module):
        def __init__(self):
                super().__init__()
                self.model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', num_classes=1)
        def forward(self, x):
                return self.model(x)['out']


    def make_model():
        model = ARCHITECTURE(ENCODER, encoder_weights=WEIGHTS, classes=1, activation=None)
        #model = DeepLabWrapper()
        return model


    test_files = sorted([x.split(".")[0] for x in os.listdir("test") if ".png" in x])
    test_ds = tu.ITSTestDataset(test_files)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)

    kf = StratifiedKFold(shuffle=True, random_state=42)

    runner = dl.SupervisedRunner(device=tu.device, 
                            input_key="image", 
                            input_target_key="mask",
                            output_key="logits")
    mask_blend = None
    iou_values = []
    
    for i, (train_inds, test_inds) in enumerate(kf.split(image_files, image_labels)):
        if i < RESUME_FOLD:
            continue
        LOGDIR = os.path.join(BASE_LOGDIR, f"fold{i}")
        gc.collect()
        torch.cuda.empty_cache()

        
        model = make_model()
        optimizer = Lookahead(RAdam(model.parameters(), lr=1e-3))
        
        scheduler = OneCycleLRWithWarmup(optimizer, num_steps=EPOCHS, lr_range=(1e-2, 1e-5), warmup_steps=1)

        train_files = image_files[train_inds]
        valid_files = image_files[test_inds]
        train_labels = [getLabel(f) for f in train_files]

        #train_ds = tu.ITSDatasetWithPL(train_files, df_pl, train_transforms=[albu.HorizontalFlip(), albu.VerticalFlip(), albu.ShiftScaleRotate()], blur_mask=False)
        train_ds = tu.ITSDataset(train_files, train_transforms=[albu.HorizontalFlip(), albu.VerticalFlip(), albu.ShiftScaleRotate()], blur_mask=False)
        
        val_ds = tu.ITSDataset(valid_files)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=6, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=6, shuffle=False)
        
        loaders = OrderedDict()
        loaders["train"] = train_loader
        loaders["valid"] = val_loader


        callbacks = [CriterionCallback(input_key="mask", output_key="logits", criterion_key="bciou", prefix="loss"),
            
            IouCallback(input_key="mask", output_key="logits",threshold=0.5),
            IouCallback(input_key="mask", output_key="logits",threshold=0.4,  prefix="iou04"),
            IouCallback(input_key="mask", output_key="logits",threshold=0.6,  prefix="iou06"),
            
            OptimizerCallback(accumulation_steps=2)

        ]
  
        if TRAINING:
            if RESUME:
                try:
                    cp = load_checkpoint(f"{LOGDIR}/checkpoints/best.pth")
                    continue
                except Exception as e:
                    pass
            runner.train(model=model, 
                    criterion=tu.criterion, 
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    loaders=loaders,
                    logdir=LOGDIR,
                    num_epochs=EPOCHS,
                    fp16=tu.fp16_params,
                    callbacks=callbacks,
                    verbose=True,
                    load_best_on_end=False,
                    resume = RESUME
                    )



        gc.collect()
        torch.cuda.empty_cache()

        model = make_model()

        cp = load_checkpoint(f"{LOGDIR}/checkpoints/best.pth")
        unpack_checkpoint(cp, model)
        model = model.eval()

        mask_fold_val = []
        for b in tqdm( runner.predict_loader(model=model, 
                                    loader=val_loader, 
                                   fp16=tu.fp16_params),
                                    total = len(test_loader)):
        
            mask_batch = torch.sigmoid(b['logits']).detach().cpu().numpy() > THRESH
            mask_batch = (mask_batch * 255).astype(np.uint8)
            mask_fold_val.extend([m.squeeze() for m in list(mask_batch)])

        for m, f in zip(mask_fold_val, valid_files):
            cv2.imwrite(os.path.join("decode_train", f"{f}.png"), m)
        gt_masks_fold = []
        gt_labels_fold = []
        for b in val_loader:
            tm = b["mask"].detach().cpu().numpy()
            gt_masks_fold.extend(list(tm))

            tl = b["label"].detach().cpu().numpy()
            gt_labels_fold.extend(list(tl))


        seg_metrics = []
        for mask, true_mask in zip(mask_fold_val, gt_masks_fold):
            seg_metrics += [np.count_nonzero(np.logical_and(true_mask, mask)) /
                    np.count_nonzero(np.logical_or(true_mask, mask))]
        print(f"Fold {i} val iou: ", np.mean(seg_metrics))
        iou_values.append(np.mean(seg_metrics))

        mask_fold = []
        for b in tqdm( runner.predict_loader(model=model, 
                                    loader=test_loader, 
                                   fp16=tu.fp16_params),
                                    total = len(test_loader)):
        
            mask_batch = torch.sigmoid(b['logits']).detach().cpu().numpy()
            mask_fold.extend(list(mask_batch))
        
        if mask_blend is None:
            mask_blend = np.array(mask_fold)
        else:
            mask_blend += np.array(mask_fold)
        
        gc.collect()
        torch.cuda.empty_cache()


    masks = [(m >  (THRESH * 5)).astype(np.uint8) * 255 for m in mask_blend]
    masks_strings = [tu.encode_mask(m) for m in masks]

    output = pd.read_csv("submission_classification.csv")

    output["base64 encoded PNG (mask)"] = masks_strings

    output.to_csv("submission_segmentation.csv", index=False)

    print("IOU CV: ", np.mean(iou_values))
        
