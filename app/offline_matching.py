import open3d as o3d
from open3d import visualization
import torch
import argparse

from react.core.react_manager import ReactManager
from react.utils.logger import getLogger
from react_embedding.embedding_net import get_embedding_model

from react.utils.viz import draw_base_dsg, draw_matching_dsg

logger = getLogger(name=__name__, log_file="offline_matching.log")


def main(args: argparse.Namespace):
    embedding_model = get_embedding_model(
        weights=args.weights,
        backbone=args.backbone,
    )

    match_threshold = args.match_threshold
    dsg_path0 = args.scene_graph_0
    dsg_path1 = args.scene_graph_1
    dsg_paths = [dsg_path0, dsg_path1]

    map_updater = ReactManager(
        match_threshold=match_threshold, embedding_model=embedding_model
    )

    viz = []
    for i, path in enumerate(dsg_paths):
        map_updater.process_dsg(
            scan_id=i,
            dsg_path=path,
        )
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
    results = map_updater.report_match_results(old_scan_id=0, new_scan_id=1)
    logger.info(map_updater)
    logger.info(f"Matches: {results.matches}")
    logger.info(f"Absent: {results.absent}")
    logger.info(f"New: {results.new}")
    logger.info(f"Travelled distance: {results.travel_distance}")
    vis = visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [1, 1, 1]
    for geo in viz:
        vis.add_geometry(geo)
    vis.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--match-threshold",
        type=float,
        default=2.5,
        help="Visual difference threshold to differentiate 2 visual embeddings",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="/home/ros/models/embeddings/iros25/embedding_lab_front.pth",
        help="The path to the embedding model's weight",
    )
    parser.add_argument(
        "-b",
        "--backbone",
        type=str,
        default="efficientnet_b2",
        help="The backbone of the embedding model",
    )
    parser.add_argument(
        "-s0",
        "--scene-graph-0",
        type=str,
        default="/home/ros/dsg_output/lab_front_1/",
        help="Path to the first scene graph",
    )
    parser.add_argument(
        "-s1",
        "--scene-graph-1",
        type=str,
        default="/home/ros/dsg_output/lab_front_2/",
        help="Path to the second scene graph",
    )
    args = parser.parse_args()
    main(args)
