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
    """
    MapUpdater class to manage instance clusters

    Attributes:
        match_threshold: distance threshold for visual embedding comparison
        embedding_model: embedding model to get visual embeddings, see
            react.net.embedding_net
        name: name of the MapUpdater object
        instance_clusters: collection of instance clusters, mapped as
            {cluster_id -> InstanceCluster}
        map_views: collection of scene images, mapped as {map_view_id -> image}
        global_instance_id: counter to generate global instance ids for Instance class
        instance_cluster_id: counter to generate cluster ids for InstanceCluster class
        include_z: whether to include z into Euclidean distance calculation for
            Hungarian matching
    """

    match_threshold: float
    embedding_model: EmbeddingNet
    name: str = "ReactUpdater"
    instance_clusters: Dict[int, InstanceCluster] = field(default_factory=dict)
    map_views: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    include_z: bool = False
    _global_instance_id: int = 0
    _instance_cluster_id: int = 0

    def __str__(self) -> str:
        updater_str = f"\n🌞 Map Updater:\n" + f"- Clusters:\n"
        for cluster in self.instance_clusters.values():
            updater_str += cluster.__str__()
        return updater_str

    def assign_instance_id(self) -> int:
        id = self._global_instance_id
        self._global_instance_id += 1
        return id

    def assign_cluster_id(self) -> int:
        id = self._instance_cluster_id
        self._instance_cluster_id += 1
        return id

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
            object_nodes[self.assign_instance_id()] = new_node
        for node in object_nodes.values():
            logger.debug(node)
        return object_nodes

    def merge_two_clusters(self, cluster_id: int, other_cluster_id: int):
        assert (
            cluster_id in self.instance_clusters
        ), f"Cluster ID is not in {self.name}: {cluster_id}"
        assert (
            other_cluster_id in self.instance_clusters
        ), f"Cluster ID is not in {self.name}: {other_cluster_id}"
        cluster: InstanceCluster = self.instance_clusters.pop(cluster_id)
        other_cluster: InstanceCluster = self.instance_clusters.pop(other_cluster_id)
        new_cluster_id = self.assign_cluster_id()
        new_instances = cluster.instances
        for inst in other_cluster.instances.values():
            new_instances.update({inst.global_id: inst})
        new_cluster = InstanceCluster(
            cluster_id=new_cluster_id, instances=new_instances
        )
        self.instance_clusters.update({new_cluster_id: new_cluster})
    #
    # def reindex_clusters(self):
    #     self.instance_cluster_id = 0
    #     new_clusters = {}
    #     for cluster in self.instance_clusters.values():
    #         cluster.cluster_id = self.instance_cluster_id
    #         new_clusters.update({self.instance_cluster_id: cluster})
    #     self.instance_clusters = new_clusters

    def init_instance_clusters(self, scan_id: int, object_nodes: Dict[int, ObjectNode]):
        for global_instance_id, node in object_nodes.items():
            instance = Instance(
                global_id=global_instance_id, node_history={scan_id: node}
            )
            instance_cluster = InstanceCluster(
                cluster_id=self.assign_cluster_id(),
                instances={global_instance_id: instance},
            )
            self.instance_clusters.update({self.assign_cluster_id(): instance_cluster})

    def optimize_cluster(self):
        while True:
            match_pair = None
            for cid, cluster in self.instance_clusters.items():
                if match_pair is not None:
                    break
                for other_cid, other_cluster in self.instance_clusters.items():
                    if (
                        cid != other_cid
                        and cluster.get_class_id() == other_cluster.get_class_id()
                    ):
                        is_match = cluster.is_match(
                            other_cluster=other_cluster,
                            match_threshold=self.match_threshold,
                        )
                        if is_match:
                            match_pair = (cid, other_cid)
                            break
            if match_pair is not None:
                assert (
                    len(match_pair) == 2
                ), f"Invalid match_pair length, needs to be 2, received {len(match_pair)}"
                self.merge_two_clusters(
                    cluster_id=match_pair[0], other_cluster_id=match_pair[1]
                )
            else:
                break
        # self.reindex_clusters()

    def update_instance_entry(self, cluster_id: int, old_inst_id, new_inst_id):
        self.instance_clusters[cluster_id].merge_two_instances(
            inst_id=old_inst_id,
            other_inst_id=new_inst_id,
            new_inst_id=self.assign_instance_id(),
        )

    def update_position_histories(self, scan_id_old: int, scan_id_new: int):
        for inst_cluster_id, inst_cluster in self.instance_clusters.items():
            matching_old_ids, matching_new_ids = inst_cluster.match_position(
                scan_id_old=scan_id_old,
                scan_id_new=scan_id_new,
                include_z=self.include_z,
            )
            for old_id, new_id in zip(matching_old_ids, matching_new_ids):
                self.update_instance_entry(
                    cluster_id=inst_cluster_id,
                    old_inst_id=old_id,
                    new_inst_id=new_id,
                )

    def get_nodes_in_scan(self, scan_id: int) -> Dict[int, ObjectNode]:
        nodes = {}
        for cluster in self.instance_clusters.values():
            for inst_id, instance in cluster.instances.items():
                if scan_id in instance.node_history:
                    node = instance.node_history[scan_id]
                    nodes.update({inst_id: node})
        return nodes

    def process_dsg(
        self,
        dsg_path: str,
        scan_id: int,
    ):
        instance_views_data, map_views_data, dsg_data = get_dsg_data(dsg_path)
        self.map_views.update({scan_id: register_map_views(map_views_data)})
        object_nodes = self.get_object_nodes_from_json_data(
            scan_id=scan_id,
            instance_views_data=instance_views_data,
            dsg_data=dsg_data,
        )
        self.init_instance_clusters(scan_id=scan_id, object_nodes=object_nodes)
        self.optimize_cluster()
