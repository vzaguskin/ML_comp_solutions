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
from catalyst.dl.callbacks import AccuracyCallback, CheckpointCallback, AUCCallback, CriterionCallback, MetricAggregationCallback, MeterMetricsCallback, VerboseLogger, SchedulerCallback, OptimizerCallback, MixupCallback
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
from efficientnet_pytorch import EfficientNet
from catalyst.contrib.nn import FocalLossMultiClass, OneCycleLRWithWarmup
import sys
if  len(sys.argv) > 1 and sys.argv[1] == "train":
    TRAINING = True
else:
    TRAINING = False
BATCH_SIZE = 8
#ENCODER = 'resnet34'

BASE_LOGDIR = './logs/catalyst_classification'
FP16 = True
DATASET_PATH = "train"
TEST_DATASET_PATH = "test"
LABELS = ['staphylococcus_epidermidis', 'klebsiella_pneumoniae', 'staphylococcus_aureus', 'moraxella_catarrhalis', 'c_kefir', 'ent_cloacae']


RESUME = None
EPOCHS = 30

image_files = [x.split(".")[0] for x in os.listdir("train") if ".png" in x]
image_files = np.array(image_files)

def getLabel(basename):
    with open(os.path.join(DATASET_PATH, basename + ".json"), 'r') as f:
            layout = json.load(f)

    label = layout['shapes'][0]['label']
    label_ind = LABELS.index(label)
    return label_ind

image_labels = [getLabel(f) for f in image_files]

def make_model():
    resnet = models.resnet34(pretrained=True)
    resnet.fc = nn.Linear(512, len(LABELS))
    return resnet

def make_model_effnet():
    return EfficientNet.from_pretrained('efficientnet-b1', num_classes=len(LABELS))

empty_mask = np.zeros((512, 640), np.uint8)
mask_string = tu.encode_mask(empty_mask)
print(mask_string)

test_files = sorted([x.split(".")[0] for x in os.listdir("test") if ".png" in x])
test_ds = tu.ITSTestDataset(test_files)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)

kf = StratifiedKFold(shuffle=True, random_state=42)

labels_blend = None
for i, (train_inds, test_inds) in enumerate(kf.split(image_files, image_labels)):
    torch.cuda.empty_cache()
    
    model = make_model()
    optimizer = Lookahead(RAdam(model.parameters(), lr=1e-3))
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.33, patience=2, verbose=True)
    scheduler = OneCycleLRWithWarmup(optimizer, num_steps=EPOCHS, lr_range=(0.003, 0.0001), warmup_steps=1)
    LOGDIR = os.path.join(BASE_LOGDIR, f"fold{i}")
    train_files = image_files[train_inds]
    valid_files = image_files[test_inds]

    train_ds = tu.ITSDataset(train_files, train_transforms=[albu.VerticalFlip(), albu.HorizontalFlip(), albu.ShiftScaleRotate(),
    albu.RandomGamma(), albu.RandomGridShuffle(), albu.HueSaturationValue()])
    val_ds = tu.ITSDataset(valid_files)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=6, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=6, shuffle=False)
    

    loaders = OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = val_loader

    runner = dl.SupervisedRunner(device=tu.device, 
                            input_key="image", 
                            input_target_key="label",
                            output_key="logits")


    callbacks = [
                CriterionCallback(input_key="label", output_key="logits", prefix="loss"),
                AccuracyCallback(input_key="label", output_key="logits", prefix="acc", activation="Sigmoid"),
                
                OptimizerCallback(accumulation_steps=2),
                #MixupCallback(alpha=0.3, input_key="label", output_key="logits", fields=("image", ))

                ]
    if TRAINING:
        runner.train(model=model, 
                    criterion=nn.CrossEntropyLoss(), 
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


    model = make_model()

    cp = load_checkpoint(f"{LOGDIR}/checkpoints/best.pth")
    unpack_checkpoint(cp, model)
    model = model.eval()

    

    labels_fold = []
    for b in tqdm( runner.predict_loader(model=model, 
                                    loader=test_loader, 
                                   fp16=tu.fp16_params),
                                    total = len(test_loader)):
    
        labels_batch = nn.functional.softmax(b['logits'], dim=1).data.cpu().numpy()
        labels_fold.extend(list(labels_batch))
    if labels_blend is None:
        labels_blend = np.array(labels_fold)
    else:
        labels_blend += np.array(labels_fold)

labels = np.argmax(labels_blend, axis=1)
label_names = [LABELS[i] for i in labels]

output = pd.DataFrame({"id":test_files,
                        "class": label_names,
                        "base64 encoded PNG (mask)": [mask_string] * len(test_files)})

output.to_csv("submission_classification.csv", index=False)
        
