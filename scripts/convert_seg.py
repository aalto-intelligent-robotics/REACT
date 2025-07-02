import os
import argparse
import cv2
from gt_config import GTCLS_2_CLS, CLS_2_COLOR
import logging
import pandas as pd
import copy
import numpy as np

GT_LABELS_FILE = "./groundtruth_labels.csv"
OUTPUT_DIR = "hydra_seg_gt"


def process_img(img_file: str, gt_dict: dict):
    img = cv2.imread(img_file)
    # img_sz = img.shape
    # for obj_id, cls_id in gt_dict.items():
    #     seg_color = copy.deepcopy(CLS_2_COLOR[GTCLS_2_CLS[cls_id]])
    #     seg_color.reverse()
    #     img[img[:, :, 0] == obj_id] = seg_color
    # print(np.unique(img))
    for obj_id in np.unique(img):
        cls_id = gt_dict[obj_id]
        cls_id_reduced = GTCLS_2_CLS[cls_id]
        seg_color = copy.deepcopy(CLS_2_COLOR[cls_id_reduced])
        seg_color.reverse()
        img[img[:,:,0] == obj_id] = seg_color
    cv2.imwrite(
        os.path.join(os.path.dirname(img_file), OUTPUT_DIR, os.path.basename(img_file)),
        img,
    )


def main(args):
    gt_df = pd.read_csv(
        GT_LABELS_FILE, usecols=["InstanceID", "ClassID"], index_col=["InstanceID"]
    )
    gt_dict = gt_df.to_dict()["ClassID"]
    print(gt_dict)

    seg_files = [
        os.path.join(args.data_path, f)
        for f in os.listdir(args.data_path)
        if "_segmentation.png" in f
    ]
    for f in seg_files:
        process_img(f, gt_dict)


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./run1")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.data_path, OUTPUT_DIR), exist_ok=True)
    main(args)
