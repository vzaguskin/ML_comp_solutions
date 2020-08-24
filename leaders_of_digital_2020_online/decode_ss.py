import os
import base64
import cv2
import pandas as pd
import numpy as np


OUTDIR = "decoded_ss_rotate"
if not os.path.exists(OUTDIR):
    os.mkdir(OUTDIR)

df = pd.read_csv("sample_submission.csv")
for i, row in df.iterrows():
    idx = row["id"]
    mask = row["base64 encoded PNG (mask)"]
    outfile = "decoded_ss_rotate/" + str(idx).zfill(3) + ".png"
    print(outfile)
    with open(outfile, 'wb') as fp:
        fp.write(base64.b64decode(mask))
    im = cv2.imread("decoded_ss_rotate/" + str(idx).zfill(3) + ".png")
    print(im.shape)
    out = cv2.imread("decoded_realsub/" + str(idx) + ".png")
    im = np.rot90(im, 3)
    im = np.fliplr(im)
    out[:, :512] = im[:512, :]
    cv2.imwrite("decoded_ss_rotate/" + str(idx).zfill(3) + ".png", out)

