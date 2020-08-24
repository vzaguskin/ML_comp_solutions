import os
import base64
import cv2
import pandas as pd
import numpy as np
from scipy import ndimage, stats
from tqdm import tqdm

REAL_DIR = "decoded_realsub_4c"
SAMPLE_DIR = "decoded_ss_rotate"
OUT_DIR = "compare_leak"
LEAK_DIR = "abuse_leak"
if not os.path.exists(LEAK_DIR):
    os.mkdir(LEAK_DIR)
EROSION_ITERATIONS = 0

def iou(true_mask, mask):
    return np.count_nonzero(np.logical_and(true_mask, mask)) / np.count_nonzero(np.logical_or(true_mask, mask))

def acc(true_mask, mask):
    return np.count_nonzero(np.logical_and(true_mask, mask))

def filterBlobs(img_sample, img_pred):
    sample_labeled = ndimage.label(img_sample)[0]
    
    res = img_pred.copy()
    for comp in ndimage.find_objects(sample_labeled):
        if not img_pred[comp].sum():
            res[comp] = img_sample[comp]
    return res

def fitBlobs(img_sample, img_pred):
    if EROSION_ITERATIONS:
        niter = EROSION_ITERATIONS
        markers = ndimage.label(cv2.erode(img_sample.copy(), None, iterations=niter))[0]
        for i in range(niter):
            markers = ndimage.maximum_filter(markers, 3)
        sample_labeled = np.zeros_like(img_sample)
        sample_labeled[img_sample > 0] = markers[img_sample > 0]
    else:
        sample_labeled = ndimage.label(img_sample)[0]

    
    #img_pred = cv2.dilate(img_pred, None)
    #img_pred = cv2.erode(img_pred, None)
    pred_labeled = ndimage.label(img_pred)[0]
    
    res = np.zeros_like(img_pred)
    for comp in ndimage.find_objects(sample_labeled):
        #print(comp[0], comp[1])
        if not img_pred[comp].sum():
            res[comp] = img_sample[comp]
        else:
            sample_center = (int(np.mean([comp[0].start,  comp[0].stop])), int(np.mean([comp[1].start,  comp[1].stop])))
            #print(sample_center)
            pred_area = pred_labeled[comp]
            pred_mode = stats.mode(pred_area[np.nonzero(pred_area)]).mode[0]
            #print(pred_mode)
            cur_pred_sclice = ndimage.find_objects(pred_labeled, max_label=pred_mode)[-1]
            pred_sample_center = (int(np.mean([cur_pred_sclice[0].start,  cur_pred_sclice[0].stop])), int(np.mean([cur_pred_sclice[1].start,  cur_pred_sclice[1].stop])))
            #print(pred_sample_center)
            dy = 0#pred_sample_center[0] - sample_center[0]
            dx = 0#pred_sample_center[1] - sample_center[1]
            best_dx = None
            best_dy = None
            best_iou = None
            best_comp = comp
            shift_limit = 6
            for dyc in range(dy - shift_limit, dy + shift_limit):
                for dxc in range(dx - shift_limit, dx + shift_limit):
                    tmp_res = np.zeros_like(img_pred)
                    slice_start_y = max(comp[0].start + dyc, 0)
                    slice_start_x = max(comp[1].start + dxc, 0)
                    slice_stop_y = max(comp[0].stop + dyc, 0)
                    slice_stop_x = max(comp[1].stop + dxc, 0)
                    target_slice = (slice(slice_start_y, slice_stop_y, None),
                            slice(slice_start_x, slice_stop_x, None))

                    source_slice = comp
                    if tmp_res[target_slice].shape == img_sample[comp].shape:
                        tmp_res[target_slice] = img_sample[comp]
                    else:
                        slice_dstart_y, slice_dstart_x, slice_dstop_y, slice_dstop_x = 0,0,0,0
                        if slice_start_y == 0:
                            slice_dstart_y = img_sample[comp].shape[0] - tmp_res[target_slice].shape[0]
                        else:
                            slice_dstop_y = tmp_res[target_slice].shape[0] - img_sample[comp].shape[0]

                        if slice_start_x == 0:
                            slice_dstart_x = img_sample[comp].shape[1] - tmp_res[target_slice].shape[1]
                        else:
                            slice_dstop_x = tmp_res[target_slice].shape[1] - img_sample[comp].shape[1]

                        source_slice = (slice(comp[0].start + slice_dstart_y, comp[0].stop + slice_dstop_y, None),
                                       slice(comp[1].start + slice_dstart_x, comp[1].stop + slice_dstop_x, None))
                        tmp_res[target_slice] = img_sample[source_slice]
                            
                        #print(img_pred, img_pred == pred_mode)
                        #print(img_pred.shape, np.count_nonzero(img_pred == 2), img_pred[0][0])
                    cur_iou = iou(tmp_res, pred_labeled == pred_mode)
                    if best_iou is None or cur_iou > best_iou:
                        best_iou = cur_iou
                        best_dx = dxc
                        best_dy = dyc
                        best_comp = source_slice
                    #else:
                    #    print("shape mismatch", tmp_res[target_slice].shape, img_sample[comp].shape)
            if best_iou is not None:
                slice_start_y = max(comp[0].start + best_dy, 0)
                slice_start_x = max(comp[1].start + best_dx, 0)
                slice_stop_y = max(comp[0].stop + best_dy, 0)
                slice_stop_x = max(comp[1].stop + best_dx, 0)
                target_slice = (slice(slice_start_y, slice_stop_y, None),
                            slice(slice_start_x, slice_stop_x, None))
                #target_slice = (slice(comp[0].start + best_dy, comp[0].stop + best_dy, None),
                 #           slice(comp[1].start + best_dx, comp[1].stop + best_dx, None))
                res[target_slice] = img_sample[best_comp]
            else:                
                print("TODO: unequal shapes")
                res[cur_pred_sclice] = img_pred[cur_pred_sclice]

            
    return res
ious = []
for f in tqdm(os.listdir(REAL_DIR)):
    im = cv2.imread(os.path.join(REAL_DIR, f), cv2.IMREAD_GRAYSCALE)
    f_filled = f.split(".")[0].zfill(3) + "." + f.split(".")[1]
    #print(f_filled)
    im2 = cv2.imread(os.path.join(SAMPLE_DIR, f_filled), cv2.IMREAD_GRAYSCALE)
    #print(im.shape, im2.shape)
    #im2[:, 512:] = 0

    imleak = im.copy()
    imsample_filtered = fitBlobs(im2[:, :512], imleak[:, :512])
    imleak[:, :512] = imsample_filtered
    cv2.imwrite(os.path.join(LEAK_DIR, f), imleak)
    ious.append(iou(imleak, im))
    c1 = im
    c2 = imleak
    c3 = np.zeros_like(c1)

    res = np.stack([c1, c2, c3], axis=2)
    #cv2.imwrite(os.path.join(OUT_DIR, f), res)
print("IOU cv: ", np.mean(ious))
