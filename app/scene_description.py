from typing import Dict, List
import yaml
import numpy as np
import cv2
import torch
import argparse
import logging
import time

from react.core.object_node import ObjectNode
from react.matching.ground_truth import GroundTruth
from react.matching.superglue_matching import SPGMatching
from react.net.embedding_net import EmbeddingNet, get_embedding_model
from react.utils.image import preprocess_image, get_instance_view
from react.utils.logger import getLogger
from react.utils.read_data import (
    get_bbox,
    get_dsg_data,
    get_node_attrs,
    register_map_views,
)

logger: logging.Logger = getLogger(name=__name__, log_file="scene_describe.log")


def get_eval_object_nodes_from_json_data(
    scan_id: int,
    instance_views_data: Dict,
    dsg_data: Dict,
) -> Dict[int, ObjectNode]:
    """
    Get object nodes for evaluation

    Args:
        scan_id: the scan_id
        instance_views_data: instance view json data
        dsg_data: dsg json data
        embedding_model: the embedding model
        map_views: the map views containing {id -> scene_image}

    Returns:
        Object nodes mapped as {node_id -> node}
    """
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
        new_node = ObjectNode(
            scan_id=scan_id,
            node_id=node_id,
            class_id=instance_data["class_id"],
            name=instance_data["name"],
            position=np.array(node_data["position"]),
            instance_views=instance_views,
            bbox=bbox,
        )
        object_nodes[node_id] = new_node
    for node in object_nodes.values():
        logger.debug(node)
    return object_nodes


def get_clusters(gt: GroundTruth, scan_id: int = 0) -> Dict[int, List[int]]:
    all_clusters = {}
    for cid, cluster in gt.matches.items():
        all_clusters.update({cid: cluster[scan_id]})
    return all_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scene-graph", default="lab_front")
    parser.add_argument("-t", "--match-threshold", default=2.2)
    parser.add_argument("-m", "--mask-threshold", action="store_true")
    parser.add_argument("-d", "--vis-dir", default="tmp")
    args = parser.parse_args()
    scene_graph_0 = args.scene_graph + "_1"
    scene_graph_1 = args.scene_graph + "_2"
    dsg_path_0 = f"/home/ros/dsg_output/{scene_graph_0}/"
    dsg_path_1 = f"/home/ros/dsg_output/{scene_graph_1}/"
    instance_views_data0, map_views_data0, dsg_data0 = get_dsg_data(dsg_path_0)
    instance_views_data1, map_views_data1, dsg_data1 = get_dsg_data(dsg_path_1)

    # Load GT data
    GT_FILE = f"../gt_matches/gt_{args.scene_graph}.yml"
    with open(GT_FILE, "r") as f:
        gt_data = yaml.safe_load(f)
    gt = GroundTruth(
        old_scan_id=0,
        new_scan_id=1,
        old_num_nodes=gt_data["num_nodes"][0],
        new_num_nodes=gt_data["num_nodes"][1],
        matches=gt_data["matches"],
    )
    all_cluster_ids_0 = get_clusters(gt=gt, scan_id=0)
    all_cluster_ids_1 = get_clusters(gt=gt, scan_id=1)

    object_nodes_0 = get_eval_object_nodes_from_json_data(
        scan_id=0,
        instance_views_data=instance_views_data0,
        dsg_data=dsg_data0,
    )
    object_nodes_1 = get_eval_object_nodes_from_json_data(
        scan_id=1,
        instance_views_data=instance_views_data1,
        dsg_data=dsg_data1,
    )
    logger.info(f"Matches: {gt.get_num_matches()}")
    logger.info(f"Absent: {gt.get_num_absent()}")
    logger.info(f"New: {gt.get_num_new()}")
    for cid, cluster in all_cluster_ids_0.items():
        logger.info(f"Cluster {cid}")
        out_str = f"{len(cluster)} "
        for node_id in cluster:
            node = object_nodes_0[node_id]
            out_str += f"{node.name}"
            break
        logger.info(out_str)


if __name__ == "__main__":
    main()
