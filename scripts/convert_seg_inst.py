import os
import argparse
import cv2
from gt_config import GTCLS_2_CLS, CLS_2_COLOR, BACKGROUND_CLS_ID
import logging
import pandas as pd
import copy
import numpy as np
import pickle

GT_LABELS_FILE = "./groundtruth_labels.csv"
OUTPUT_SEG_DIR = "hydra_inst"
OUTPUT_MASK_DIR = "hydra_inst/mask"


def process_img(img_file: str, gt_dict: dict):
    img = cv2.imread(img_file)
    inst_ids = np.unique(img)
    mask = {}
    # np.zeros([len(inst_ids), img.shape[0], img.shape[1]], dtype=np.uint8)
    for i, inst_id in enumerate(inst_ids):
        cls_id = gt_dict[inst_id]
        if cls_id in BACKGROUND_CLS_ID:
            seg_color = [0.0, 0.0, 0.0]
        else:
            seg_color = copy.deepcopy(CLS_2_COLOR[cls_id])
            seg_color.reverse()
            mask[i] = {"class_id": cls_id - 2, "mask": img[:, :, 0] == inst_id}
        img[img[:, :, 0] == inst_id] = seg_color
    cv2.imwrite(
        os.path.join(
            os.path.dirname(img_file), OUTPUT_SEG_DIR, os.path.basename(img_file)
        ),
        img,
    )
    with open(
        os.path.join(
            os.path.dirname(img_file),
            OUTPUT_MASK_DIR,
            os.path.basename(img_file).replace("segmentation.png", "masks.pkl"),
        ),
        "wb",
    ) as f:
        pickle.dump(mask, f)


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
    os.makedirs(os.path.join(args.data_path, OUTPUT_SEG_DIR), exist_ok=True)
    main(args)
