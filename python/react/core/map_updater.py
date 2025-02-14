import logging
import numpy as np
from typing import Dict, List, Set
from dataclasses import dataclass, field
import torch
from torch import Tensor
import cv2

from react.core.object_node import ObjectNode
from react.core.instance import Instance
from react.core.instance_cluster import InstanceCluster
from react.net.embedding_net import EmbeddingNet
from react.utils.logger import getLogger
from react.utils.read_data import (
    get_bbox,
    get_dsg_data,
    get_node_attrs,
    register_map_views,
)
from react.utils.image import get_instance_view, preprocess_image

logger: logging.Logger = getLogger(name=__name__, log_file="map_updater.log")


@dataclass
class MapUpdater:
    match_threshold: float
    embedding_model: EmbeddingNet
    name: str = "ReactUpdater"
    instance_clusters: Dict[int, InstanceCluster] = field(default_factory=dict)
    map_views: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    global_instance_id: int = 0
    instance_cluster_id: int = 0
    include_z: bool = False

    def get_object_nodes_from_json_data(
        self,
        scan_id: int,
        instance_views_data: Dict,
        dsg_data: Dict,
    ) -> Dict[int, ObjectNode]:
        object_nodes: Dict[int, ObjectNode] = {}
        num_features = self.embedding_model.num_features()
        scan_map_views = self.map_views[scan_id]
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
            view_embeddings = torch.zeros(1, num_features)
            for i, mask_data in enumerate(all_masks_data):
                mask_file = mask_data["file"]
                map_view_id = mask_data["map_view_id"]
                mask = cv2.imread(mask_file)[:, :, 0]
                instance_views[map_view_id] = mask
                view_img = get_instance_view(
                    map_view_img=scan_map_views[map_view_id],
                    mask=mask,
                    mask_bg=True,
                    crop=True,
                    padding=10,
                )
                preprocessed_img = preprocess_image(view_img)
                embedding = self.embedding_model(preprocessed_img).detach().cpu()
                view_embeddings += embedding
            view_embeddings /= len(all_masks_data)
            new_node = ObjectNode(
                scan_id=scan_id,
                node_id=node_id,
                class_id=instance_data["class_id"],
                name=instance_data["name"],
                position=np.array(node_data["position"]),
                instance_views=instance_views,
                bbox=bbox,
                embedding=view_embeddings,
            )
            self.global_instance_id += 1
            object_nodes[self.global_instance_id] = new_node
        for node in object_nodes.values():
            logger.debug(node)
        return object_nodes

    def init_instance_clusters(self, scan_id: int, object_nodes: Dict[int, ObjectNode]):
        for global_instance_id, node in object_nodes.items():
            instance = Instance(
                global_id=global_instance_id, node_history={scan_id: node}
            )
            instance_cluster = InstanceCluster(
                cluster_id=self.instance_cluster_id,
                instances={global_instance_id: instance},
            )
            self.instance_cluster_id += 1
            self.instance_clusters.update({self.instance_cluster_id: instance_cluster})

    # def cluster(self, object_nodes: Dict[int, ObjectNode]):
    #     for global_instance_id, node in object_nodes.items():
    #         pass

    def process_dsg(
        self,
        dsg_path: str,
        scan_id: int,
        # save_path: str = "./output/",
        # save_json: bool = True,
        # re_cluster: bool = False,
    ):
        instance_views_data, map_views_data, dsg_data = get_dsg_data(dsg_path)
        self.map_views.update({scan_id: register_map_views(map_views_data)})
        object_nodes = self.get_object_nodes_from_json_data(
            scan_id=scan_id,
            instance_views_data=instance_views_data,
            dsg_data=dsg_data,
        )
        self.init_instance_clusters(scan_id=scan_id, object_nodes=object_nodes)
        # self.cluster(dsg_id=scan_id)
        # if re_cluster:
        #     self.re_cluster()
        # if save_json:
        #     save_instance_sets(
        #         instance_sets=self.instance_sets,
        #         fname=f"{save_path}/instance_sets{scan_id}.json",
        #     )
