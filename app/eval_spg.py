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

logger: logging.Logger = getLogger(name=__name__, log_file="eval_spg.log")


def are_same_nodes_spg(
    node: ObjectNode,
    other_node: ObjectNode,
    matcher,
    map_views: Dict[int, np.ndarray],
    mask_bg: bool,
):
    start = time.perf_counter()
    for map_view_id, mask in node.instance_views.items():
        view0 = get_instance_view(
            map_view_img=map_views[map_view_id],
            mask=mask,
            mask_bg=mask_bg,
            crop=True,
            padding=10,
        )
        for other_map_view_id, other_mask in other_node.instance_views.items():
            view1 = get_instance_view(
                map_view_img=map_views[other_map_view_id],
                mask=other_mask,
                mask_bg=mask_bg,
                crop=True,
                padding=10,
            )
            _, _, matches, confidence = matcher(
                view0,
                view1,
                filename=f"{node.name}_{node.node_id}-{other_node.name}_{other_node.node_id}.png",
            )
            match_score = matcher.get_confidence_score(
                matches=matches, confidences=confidence
            )
            if match_score > matcher.match_conf_threshold:
                end = time.perf_counter()
                elapsed = end - start
                logger.info(
                    f"Conf: {match_score}, Elapsed time: {elapsed}; FPS: {1/elapsed}"
                )
                return True
    return False


def evaluate(
    all_clusters: Dict[int, List[int]],
    all_object_nodes: Dict[int, ObjectNode],
    matcher: SPGMatching,
    map_views: Dict[int, np.ndarray],
    mask_bg: bool,
):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    matched_pairs = []
    for cid, cluster in all_clusters.items():
        for node_id in cluster:
            node = all_object_nodes[node_id]
            for other_cid, other_cluster in all_clusters.items():
                for other_node_id in other_cluster:
                    if other_node_id == node_id:
                        continue
                    other_node = all_object_nodes[other_node_id]
                    if node.class_id != other_node.class_id:
                        continue
                    if (node.node_id, other_node.node_id) in matched_pairs:
                        continue
                    if (other_node.node_id, node.node_id) in matched_pairs:
                        continue
                    matched_pairs.append((node.node_id, other_node.node_id))
                    # Same objects
                    if other_cid == cid:
                        if are_same_nodes_spg(
                            node=node,
                            other_node=other_node,
                            matcher=matcher,
                            map_views=map_views,
                            mask_bg=mask_bg,
                        ):
                            tp += 1
                        else:
                            fn += 1
                    # Different objects
                    else:
                        if are_same_nodes_spg(
                            node=node,
                            other_node=other_node,
                            matcher=matcher,
                            map_views=map_views,
                            mask_bg=mask_bg,
                        ):
                            fp += 1
                        else:
                            tn += 1

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def get_eval_object_nodes_from_json_data(
    scan_id: int,
    instance_views_data: Dict,
    dsg_data: Dict,
    embedding_model: EmbeddingNet,
    map_views: Dict[int, np.ndarray],
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
    num_features = embedding_model.num_features()
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
        view_embeddings = torch.zeros(len(all_masks_data), num_features)
        for i, mask_data in enumerate(all_masks_data):
            mask_file = mask_data["file"]
            map_view_id = mask_data["map_view_id"]
            mask = cv2.imread(mask_file)[:, :, 0]
            instance_views[map_view_id] = mask
            view_img = get_instance_view(
                map_view_img=map_views[map_view_id],
                mask=mask,
                mask_bg=True,
                crop=True,
                padding=10,
            )
            preprocessed_img = preprocess_image(view_img)
            embedding = embedding_model(preprocessed_img).detach().cpu()
            view_embeddings[i, :] = embedding
        new_node = ObjectNode(
            scan_id=scan_id,
            node_id=node_id,
            class_id=instance_data["class_id"],
            name=instance_data["name"],
            position=np.array(node_data["position"]),
            instance_views=instance_views,
            bbox=bbox,
            embedding=torch.quantile(view_embeddings, q=0.5, dim=0),
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
    scene_graph = args.scene_graph
    match_thresholds = args.match_threshold
    dsg_path = f"/home/ros/dsg_output/{scene_graph}_2/"
    embedding_model = get_embedding_model(
        weights=f"/home/ros/models/embeddings/iros25/embedding_{scene_graph}.pth",
        backbone="efficientnet_b2",
    )
    instance_views_data, map_views_data, dsg_data = get_dsg_data(dsg_path)

    # Load GT data
    GT_FILE = f"../gt_matches/gt_{scene_graph}.yml"
    with open(GT_FILE, "r") as f:
        gt_data = yaml.safe_load(f)
    gt = GroundTruth(
        old_scan_id=0,
        new_scan_id=1,
        old_num_nodes=gt_data["num_nodes"][0],
        new_num_nodes=gt_data["num_nodes"][1],
        matches=gt_data["matches"],
    )
    all_cluster_ids = get_clusters(gt=gt, scan_id=0)
    logger.info(all_cluster_ids)

    map_views = register_map_views(map_views_data=map_views_data)
    object_nodes = get_eval_object_nodes_from_json_data(
        scan_id=0,
        instance_views_data=instance_views_data,
        dsg_data=dsg_data,
        embedding_model=embedding_model,
        map_views=map_views,
    )
    matcher = SPGMatching(
        default_vis_dir=args.vis_dir, print_images=True, only_print_match=True
    )
    results = evaluate(
        all_clusters=all_cluster_ids,
        all_object_nodes=object_nodes,
        matcher=matcher,
        map_views=map_views,
        mask_bg=args.mask_threshold,
    )
    logger.info(results)


if __name__ == "__main__":
    main()
