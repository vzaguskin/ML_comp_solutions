import os
import base64
import cv2
import pandas as pd
import numpy as np

OUTDIR = "decoded_realsub"
if not os.path.exists(OUTDIR):
    os.mkdir(OUTDIR)

df = pd.read_csv("submission_segmentation.csv")
for i, row in df.iterrows():
    idx = row["id"]
    mask = row["base64 encoded PNG (mask)"]
    outfile = "decoded_realsub/" + str(idx) + ".png"
    print(outfile)
    with open(outfile, 'wb') as fp:
        fp.write(base64.b64decode(mask))
    

