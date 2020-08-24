import os
import base64
import cv2
import pandas as pd
import numpy as np

output = pd.read_csv("submission_classification.csv").set_index("id")

clf_keys = output["class"].unique()
key_dict = dict(zip(clf_keys, np.roll(clf_keys, 1)))
print(key_dict)

for f in os.listdir("abuse_leak"):
    with open(os.path.join("abuse_leak", f), "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    idx = int(f.split(".")[0])
    print(idx, output.loc[idx, "class"])
    output.loc[idx, "base64 encoded PNG (mask)"] = encoded_string
    #output.loc[idx, "class"] = key_dict[output.loc[idx, "class"]]

output = output.reset_index()

output.to_csv("submission_seg_leaky.csv", index=False)
    
