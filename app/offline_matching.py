import open3d as o3d
from open3d import visualization

from react.core.map_updater import MapUpdater
from react.net.embedding_net import get_embedding_model
from react.utils.logger import getLogger

# from react.utils.viz import draw_base_dsg, draw_matching_dsg

logger = getLogger(name=__name__, log_file="offline_matching.log")

"""
TODO: Create yaml file for all params here
Params
- COFFEE ROOM: 
  weights: "/home/ros/models/embeddings/iros25/embedding_coffee_room.pth"
  backbone: "efficientnet_b2"
  match_threshold: 2.0
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
    embedding_model = get_embedding_model(
        weights="/home/ros/models/embeddings/iros25/embedding_coffee_room.pth",
        backbone="efficientnet_b2",
    )

    SAVE_PATH = "./output/"
    MATCH_THRESHOLD = 2.0
    DSG_PATH0 = "/home/ros/dsg_output/coffee_room_1/"
    DSG_PATH1 = "/home/ros/dsg_output/coffee_room_2/"
    dsg_paths = [DSG_PATH0, DSG_PATH1]

    map_updater = MapUpdater(
        match_threshold=MATCH_THRESHOLD, embedding_model=embedding_model
    )

    viz = []
    for i, path in enumerate(dsg_paths):
        map_updater.process_dsg(
            scan_id=i,
            dsg_path=path,
            re_cluster=True,
        )
        # mesh = o3d.io.read_triangle_mesh(f"{path}/backend/mesh.ply")
        # if i == 0:
        #     dsg_viz = draw_base_dsg(
        #         dsg_id=i,
        #         mesh=mesh,
        #         map_updater=map_updater,
        #         node_label_z=3,
        #         set_label_z=5,
        #         dsg_offset_z=0,
        #         include_scene_mesh=True,
        #         include_instance_mesh=False,
        #     )
        # else:
        # map_updater.update_position_histories(dsg_id_old=i - 1, dsg_id_new=i)
        # dsg_viz = draw_matching_dsg(
        #     dsg_id_old=i - 1,
        #     dsg_id_new=i,
        #     mesh=mesh,
        #     map_updater=map_updater,
        #     old_dsg_offset_z=-5 * (i - 1),
        #     new_dsg_offset_z=-5 * i,
        #     include_scene_mesh=True,
        #     include_instance_mesh=False,
        # )
    #     viz += dsg_viz
    # total_dist = map_updater.report_travel_dist()
    # logger.info(f"Total travel dist: {total_dist}")
    # # visualization.draw(viz)
    # vis = visualization.Visualizer()
    # vis.create_window(visible=True)
    # # Call only after creating visualizer window.
    # vis.get_render_option().background_color = [1, 1, 1]
    # # vis.add_geometry(pcd)
    # for geo in viz:
    #     vis.add_geometry(geo)
    # vis.run()
