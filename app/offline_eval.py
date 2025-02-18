from copy import deepcopy
import open3d as o3d
from open3d import visualization
import torch
import numpy as np
import yaml
import argparse

from react.core.map_updater import MapUpdater
from react.matching.ground_truth import GroundTruth
from react.matching.match_results import MatchResults
from react.eval.evaluator import ReactEvaluator
from react.net.embedding_net import get_embedding_model
from react.utils.logger import getLogger
from react.utils.viz import draw_base_dsg, draw_matching_dsg

logger = getLogger(name=__name__, log_file="offline_matching.log")

"""
TODO: Create yaml file for all params here
Params
- COFFEE ROOM: 
  weights: "/home/ros/models/embeddings/iros25/embedding_coffee_room_2.pth"
  backbone: "efficientnet_b2"
  match_threshold: 2.5
- FLAT:
  weights: "/home/ros/models/embeddings/iros25/embedding_flat.pth"
  backbone: "efficientnet_b2"
  match_threshold: 1.0
- LAB FRONT:
  weights: "/home/ros/models/embeddings/iros25/embedding_lab_front.pth"
  backbone: "efficientnet_b2"
  match_threshold: 2.0
- STUDY HALL:
  weights: "/home/ros/models/embeddings/iros25/embedding_study_hall.pth"
  backbone: "efficientnet_b2"
  match_threshold: 2.2
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scene-graph", default="lab_front")
    args = parser.parse_args()
    SCENE_GRAPH = args.scene_graph
    MATCH_THRESHOLD = np.arange(start=0.0, stop=10.1, step=0.2)
    embedding_model = get_embedding_model(
        weights=f"/home/ros/models/embeddings/iros25/embedding_{SCENE_GRAPH}.pth",
        backbone="efficientnet_b2",
    )

    SAVE_PATH = "./output/"
    DSG_PATH0 = f"/home/ros/dsg_output/{SCENE_GRAPH}_1/"
    DSG_PATH1 = f"/home/ros/dsg_output/{SCENE_GRAPH}_2/"
    GT_FILE = f"../gt_mathces/gt_{SCENE_GRAPH}.yml"

    # Load GT data
    OLD_SCAN_ID = 0
    NEW_SCAN_ID = 1
    with open(GT_FILE, "r") as f:
        gt_data = yaml.safe_load(f)
    gt = GroundTruth(
        old_scan_id=OLD_SCAN_ID,
        new_scan_id=NEW_SCAN_ID,
        old_num_nodes=gt_data["num_nodes"][OLD_SCAN_ID],
        new_num_nodes=gt_data["num_nodes"][NEW_SCAN_ID],
        matches=gt_data["matches"],
    )
    logger.info(f"Num matches: {gt.get_num_matches()}")
    logger.info(f"Num absent: {gt.get_num_absent()}")
    logger.info(f"Num new: {gt.get_num_new()}")

    # Map updater
    dsg_paths = [DSG_PATH0, DSG_PATH1]
    map_updater = MapUpdater(
        match_threshold=MATCH_THRESHOLD[0], embedding_model=embedding_model
    )

    viz = []
    for i, path in enumerate(dsg_paths):
        map_updater.process_dsg(
            scan_id=i,
            dsg_path=path,
            optimize_cluster=False,
        )
    ref_map_updater = deepcopy(map_updater)
    for thresh in MATCH_THRESHOLD:
        logger.info(f"Match threshold: {thresh}")
        map_updater = deepcopy(ref_map_updater)
        map_updater.match_threshold = thresh
        for i, path in enumerate(dsg_paths):
            map_updater.optimize_cluster()
            mesh = o3d.io.read_triangle_mesh(f"{path}/backend/mesh.ply")
            if i == 0:
                dsg_viz = draw_base_dsg(
                    scan_id=i,
                    mesh=mesh,
                    map_updater=map_updater,
                    node_label_z=3,
                    set_label_z=5,
                    dsg_offset_z=0,
                    include_scene_mesh=True,
                    include_instance_mesh=False,
                )
            else:
                map_updater.update_position_histories(scan_id_old=i - 1, scan_id_new=i)
                dsg_viz = draw_matching_dsg(
                    scan_id_old=i - 1,
                    scan_id_new=i,
                    mesh=mesh,
                    map_updater=map_updater,
                    old_dsg_offset_z=-5 * (i - 1),
                    new_dsg_offset_z=-5 * i,
                    include_scene_mesh=True,
                    include_instance_mesh=False,
                )
            viz += dsg_viz
        results: MatchResults = map_updater.report_match_results(
            old_scan_id=0, new_scan_id=1
        )
        evaluator = ReactEvaluator(ground_truth=gt, match_results=results)
        evaluator.count_matched_results()
        match_pr = evaluator.get_precision_recall()
        # new_eval_cnt = evaluator.count_new_results()
        # absent_eval_cnt = evaluator.count_absent_results()
        # new_pr = evaluator.get_precision_recall(new_eval_cnt)
        # absent_pr = evaluator.get_precision_recall(absent_eval_cnt)
        logger.info(f"Match Pre: {match_pr['m']['pre']}")
        logger.info(f"Match Rec: {match_pr['m']['rec']}")
        logger.info(f"Absent Pre: {match_pr['a']['pre']}")
        logger.info(f"Absent Rec: {match_pr['a']['rec']}")
        logger.info(f"New Pre: {match_pr['n']['pre']}")
        logger.info(f"New Rec: {match_pr['n']['rec']}")
        # logger.info(f"New TP: {new_eval_cnt['tp']}")
        # logger.info(f"New FP: {new_eval_cnt['fp']}")
        # logger.info(f"New FN: {new_eval_cnt['fn']}")
        # logger.info(f"New Pre: {new_pr['precision']}")
        # logger.info(f"New Rec: {new_pr['recall']}")
        # logger.info(f"Absent Pre: {absent_pr['precision']}")
        # logger.info(f"Absent Rec: {absent_pr['recall']}")
        logger.debug(f"Map at threshold {thresh}:\n{map_updater}")
        visualization.draw(viz)
