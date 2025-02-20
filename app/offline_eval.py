from copy import deepcopy
from typing import List
import open3d as o3d
import os
from open3d import visualization
import torch
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt

from react.core.react_manager import ReactManager
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


def run(
    m_thresh_f1: List,
    a_thresh_f1: List,
    n_thresh_f1: List,
    all_thresh_f1: List,
    travel_dist: List,
    viz: List,
    match_thresholds: np.ndarray,
    greedy: bool,
    ref_map_updater: ReactManager,
):
    for thresh in match_thresholds:
        map_updater = deepcopy(ref_map_updater)
        map_updater.match_threshold = thresh
        for i, path in enumerate(dsg_paths):
            if not greedy:
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
                if not greedy:
                    map_updater.update_position_histories(
                        scan_id_old=i - 1, scan_id_new=i
                    )
                else:
                    map_updater.greedy_match(scan_id_old=i - 1, scan_id_new=i)
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
        match_metrics = evaluator.get_metrics()
        m_thresh_f1.append([thresh, match_metrics["m"]["f1"]])
        a_thresh_f1.append([thresh, match_metrics["a"]["f1"]])
        n_thresh_f1.append([thresh, match_metrics["n"]["f1"]])
        all_thresh_f1.append([thresh, match_metrics["all"]["f1"]])
        travel_dist.append([thresh, results.travel_distance])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scene-graph", default="lab_front")
    # parser.add_argument("-g", "--greedy", action="store_true")
    args = parser.parse_args()
    SCENE_GRAPH = args.scene_graph
    MATCH_THRESHOLDS = np.arange(start=0.0, stop=5.1, step=0.2)
    embedding_model = get_embedding_model(
        weights=f"/home/ros/models/embeddings/iros25/embedding_{SCENE_GRAPH}.pth",
        backbone="efficientnet_b2",
    )

    SAVE_PATH = "./output/"
    DSG_PATH0 = f"/home/ros/dsg_output/{SCENE_GRAPH}_1/"
    DSG_PATH1 = f"/home/ros/dsg_output/{SCENE_GRAPH}_2/"
    GT_FILE = f"../gt_matches/gt_{SCENE_GRAPH}.yml"

    # Load GT data
    OLD_SCAN_ID = 0
    NEW_SCAN_ID = 1

    # Metrics storage
    m_thresh_f1 = []
    a_thresh_f1 = []
    n_thresh_f1 = []
    all_thresh_f1 = []
    m_thresh_f1_greedy = []
    a_thresh_f1_greedy = []
    n_thresh_f1_greedy = []
    all_thresh_f1_greedy = []
    travel_dist = []
    travel_dist_greedy = []

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
    map_updater = ReactManager(
        match_threshold=MATCH_THRESHOLDS[0], embedding_model=embedding_model
    )

    viz = []
    for i, path in enumerate(dsg_paths):
        map_updater.process_dsg(
            scan_id=i,
            dsg_path=path,
            optimize_cluster=False,
        )
    ref_map_updater = deepcopy(map_updater)
    # Run REACT
    run(
        m_thresh_f1=m_thresh_f1,
        a_thresh_f1=a_thresh_f1,
        n_thresh_f1=n_thresh_f1,
        all_thresh_f1=all_thresh_f1,
        travel_dist=travel_dist,
        viz=viz,
        match_thresholds=MATCH_THRESHOLDS,
        greedy=False,
        ref_map_updater=ref_map_updater,
    )
    # Run 1v1 Greedy matching
    run(
        m_thresh_f1=m_thresh_f1_greedy,
        a_thresh_f1=a_thresh_f1_greedy,
        n_thresh_f1=n_thresh_f1_greedy,
        all_thresh_f1=all_thresh_f1_greedy,
        travel_dist=travel_dist_greedy,
        viz=viz,
        match_thresholds=MATCH_THRESHOLDS,
        greedy=True,
        ref_map_updater=ref_map_updater,
    )
    m_thresh_f1 = np.array(m_thresh_f1)
    a_thresh_f1 = np.array(a_thresh_f1)
    n_thresh_f1 = np.array(n_thresh_f1)
    all_thresh_f1 = np.array(all_thresh_f1)
    travel_dist = np.array(travel_dist)
    m_thresh_f1_greedy = np.array(m_thresh_f1_greedy)
    a_thresh_f1_greedy = np.array(a_thresh_f1_greedy)
    n_thresh_f1_greedy = np.array(n_thresh_f1_greedy)
    all_thresh_f1_greedy = np.array(all_thresh_f1_greedy)
    travel_dist_greedy = np.array(travel_dist_greedy)
    os.makedirs(f"./exp_outputs/{args.scene_graph}", exist_ok=True)
    np.save(f"./exp_outputs/{args.scene_graph}/m_thresh_f1.npy", m_thresh_f1)
    np.save(f"./exp_outputs/{args.scene_graph}/a_thresh_f1.npy", a_thresh_f1)
    np.save(f"./exp_outputs/{args.scene_graph}/n_thresh_f1.npy", n_thresh_f1)
    np.save(f"./exp_outputs/{args.scene_graph}/all_thresh_f1.npy", all_thresh_f1)
    np.save(f"./exp_outputs/{args.scene_graph}/travel_dist.npy", travel_dist)
    np.save(f"./exp_outputs/{args.scene_graph}/m_thresh_f1_greedy.npy", m_thresh_f1_greedy)
    np.save(f"./exp_outputs/{args.scene_graph}/a_thresh_f1_greedy.npy", a_thresh_f1_greedy)
    np.save(f"./exp_outputs/{args.scene_graph}/n_thresh_f1_greedy.npy", n_thresh_f1_greedy)
    np.save(f"./exp_outputs/{args.scene_graph}/all_thresh_f1_greedy.npy", all_thresh_f1_greedy)
    np.save(f"./exp_outputs/{args.scene_graph}/travel_dist_greedy.npy", travel_dist_greedy)
    fig, ax = plt.subplots(5)
    ax[0].plot(m_thresh_f1[:, 0], m_thresh_f1[:, 1], label='REACT')
    ax[0].plot(m_thresh_f1_greedy[:, 0], m_thresh_f1_greedy[:, 1], label='w/o cluster')
    ax[1].plot(a_thresh_f1[:, 0], a_thresh_f1[:, 1], label='REACT')
    ax[1].plot(a_thresh_f1_greedy[:, 0], a_thresh_f1_greedy[:, 1], label='w/o cluster')
    ax[2].plot(n_thresh_f1[:, 0], n_thresh_f1[:, 1], label='REACT')
    ax[2].plot(n_thresh_f1_greedy[:, 0], n_thresh_f1_greedy[:, 1], label='w/o cluster')
    ax[3].plot(all_thresh_f1[:, 0], all_thresh_f1[:, 1], label='REACT')
    ax[3].plot(all_thresh_f1_greedy[:, 0], all_thresh_f1_greedy[:, 1], label='w/o cluster')
    ax[4].plot(travel_dist_greedy[:, 0], travel_dist[:, 1], label='REACT')
    ax[4].plot(travel_dist_greedy[:, 0], travel_dist_greedy[:, 1], label='w/o cluster')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    ax[4].legend()
    fig.suptitle(SCENE_GRAPH)
    ax[0].title.set_text("Matched")
    ax[1].title.set_text("Absent")
    ax[2].title.set_text("New")
    ax[3].title.set_text("All")
    ax[4].title.set_text("Travel distance")
    plt.show()
