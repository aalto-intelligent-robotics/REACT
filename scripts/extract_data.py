import json
import os
from pathlib import Path
import random
import shutil
from tabnanny import check
import cv2
import logging
from typing import Dict, List, Union
import numpy as np
import time
import argparse

from instance_matching.core.object_nodes import ObjectNode
from instance_matching.utils.read_data import get_bbox


def get_node_attrs(dsg_data, node_id) -> Dict:
    for node_data in dsg_data["nodes"]:
        if node_data["id"] == node_id:
            return node_data["attributes"]
    return {}


def get_instance_view(
    map_view_img: np.ndarray,
    mask,
    mask_bg: bool = True,
    crop: bool = True,
    padding: int = 5,
):
    coords = cv2.findNonZero(mask)
    # Get bounding box (x, y, width, height)
    x, y, w, h = cv2.boundingRect(coords)
    # Crop the image using the bounding box
    if mask_bg:
        image = cv2.bitwise_and(map_view_img, map_view_img, mask=mask)
    else:
        image = map_view_img
    if crop:
        image = image[
            max(y - padding, 0) : min(y + padding + h, map_view_img.shape[0]),
            max(x - padding, 0) : min(x + padding + w, map_view_img.shape[1]),
        ]
    return image


def register_hydra_nodes_from_json_data(
    instance_views_data: Dict,
    dsg_data: Dict,
    map_views: Dict[int, np.ndarray],
) -> Dict[int, ObjectNode]:
    global_instance_id = 0
    object_nodes: Dict[int, ObjectNode] = {}
    for instance_data in instance_views_data:
        node_id = instance_data["node_id"]
        all_masks_data = instance_data["masks"]
        node_data = get_node_attrs(dsg_data, node_id)
        bbox_data = node_data["bounding_box"]
        bbox = get_bbox(
            dimensions=bbox_data["dimensions"], position=bbox_data["world_P_center"]
        )
        assert node_data is not None, f"{node_id} not found from dsg"
        instance_views = {}
        for _, mask_data in enumerate(all_masks_data):
            mask_file = mask_data["file"]
            map_view_id = mask_data["map_view_id"]
            mask = cv2.imread(mask_file)[:, :, 0]
            view_img = get_instance_view(
                map_view_img=map_views[map_view_id],
                mask=mask,
                mask_bg=True,
                crop=True,
                padding=10,
            )
            instance_views[map_view_id] = view_img
        new_node = ObjectNode(
            global_instance_id=global_instance_id,
            node_id=node_id,
            class_id=instance_data["class_id"],
            name=instance_data["name"],
            position=np.array(node_data["position"]),
            instance_views=instance_views,
            mesh_connections=set(node_data["mesh_connections"]),
            bbox=bbox,
        )
        object_nodes[global_instance_id] = new_node
        global_instance_id += 1
    for node in object_nodes.values():
        logging.debug(
            f"Node id {node.node_id} instance_views size {len(node.instance_views)}"
        )
    return object_nodes


def register_map_views(map_views_data) -> Dict[int, np.ndarray]:
    map_views = {}
    for view_data in map_views_data:
        map_view_file = view_data["file"]
        map_view_id = view_data["map_view_id"]
        map_views[map_view_id] = cv2.imread(map_view_file)
    return map_views


def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0] for img in img_list)

    # image resizing
    im_list_resize = [
        cv2.resize(
            img,
            (int(img.shape[1] * h_min / img.shape[0]), h_min),
            interpolation=interpolation,
        )
        for img in img_list
    ]

    # return final image
    return cv2.hconcat(im_list_resize)


def vconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1] for img in img_list)

    # resizing images
    im_list_resize = [
        cv2.resize(
            img,
            (w_min, int(img.shape[0] * w_min / img.shape[1])),
            interpolation=interpolation,
        )
        for img in img_list
    ]
    # return final image
    return cv2.vconcat(im_list_resize)


def are_same_objects(node: ObjectNode, other_node: ObjectNode):
    img = []
    other_img = []
    for i in range(3):
        img.append(random.choice(list(node.instance_views.values())))
        other_img.append(random.choice(list(other_node.instance_views.values())))
    img_win = vconcat_resize([hconcat_resize(img), hconcat_resize(other_img)])
    cv2.imshow("img_cmp", img_win)
    while cv2.waitKey(0) != ord("q"):
        pass
    cv2.destroyAllWindows()
    print("Were those objects similar [y/n]?")
    ans = input()
    while ans not in ["y", "n"]:
        print("Answer either y/n")
        ans = input()
    return True if ans == "y" else False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scene_graph", type=str, default="flat1")
    opts = parser.parse_args()
    logging.basicConfig(
        format="[%(asctime)s %(filename)s][%(levelname)s]: %(message)s",
        level=logging.INFO,
    )

    DSG_PATH = "/home/ros/dsg_output/"
    SCENE_GRAPH = opts.scene_graph
    with open(f"{DSG_PATH}/{SCENE_GRAPH}/instance_views/instance_views.json") as f:
        instance_views_data = json.load(f)
    with open(f"{DSG_PATH}/{SCENE_GRAPH}/map_views/map_views.json") as f:
        map_views_data = json.load(f)
    with open(f"{DSG_PATH}/{SCENE_GRAPH}/backend/dsg.json") as f:
        dsg_data = json.load(f)

    # {map_view_id -> image}
    map_views = register_map_views(map_views_data)
    # Has instance_views: {map_view_id -> mask}
    hydra_nodes = register_hydra_nodes_from_json_data(
        instance_views_data, dsg_data, map_views
    )
    OUTPUT_DIR = "./instance_views/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    instance_sets: Dict[int, List[np.ndarray]] = {}
    checked_ids = set()
    for node in hydra_nodes.values():
        if node.node_id not in checked_ids:
            checked_ids.add(node.node_id)
            instance_sets[node.node_id] = list(node.instance_views.values())
            for other_node in [
                n
                for n in hydra_nodes.values()
                if n.class_id == node.class_id and n.node_id not in checked_ids
            ]:
                if are_same_objects(node, other_node):
                    instance_sets[node.node_id] += list(other_node.instance_views.values())
                    checked_ids.add(other_node.node_id)
    for id, views in instance_sets.items():
        instance_dir = Path(OUTPUT_DIR) / Path(str(id))
        os.makedirs(instance_dir, exist_ok=True)
        for v_id, v in enumerate(views):
            filename = instance_dir / Path(f"{str(v_id).zfill(5)}.png")
            cv2.imwrite(str(filename), v)


if __name__ == "__main__":
    main()
