import os
import base64
import cv2
import pandas as pd
import numpy as np

OUTDIR = "decoded_realsub_aligned"
if not os.path.exists(OUTDIR):
    os.mkdir(OUTDIR)
df = pd.read_csv("submission_seg_leaky.csv")
for i, row in df.iterrows():
    idx = row["id"]
    mask = row["base64 encoded PNG (mask)"]
    outfile = os.path.join(OUTDIR, str(idx).zfill(3) + ".png")
    print(outfile)
    with open(outfile, 'wb') as fp:
        fp.write(base64.b64decode(mask))
