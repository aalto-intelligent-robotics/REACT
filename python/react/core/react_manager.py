from copy import deepcopy
import logging
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field
from react.core import instance
import torch
from torch import Tensor
import cv2

from react.core.object_node import ObjectNode
from react.core.instance import Instance
from react.core.instance_cluster import InstanceCluster
from react_embedding.embedding_net import EmbeddingNet
from react.matching.match_results import MatchResults
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
class ReactManager:
    """ReactManager class to manage instance clusters.

    :param visual_difference_threshold: Visual difference threshold for
        visual embedding comparison.
    :param embedding_model: Embedding model to get visual embeddings.
    :param name: Name of the ReactManager object.
    :param include_z: Whether to include z axis into Euclidean distance
        calculation for Hungarian matching.
    :param _instance_clusters: Collection of instance clusters, mapped
        as {cluster_id -> InstanceCluster}.
    :param _map_views: Collection of scene images, mapped as
        {map_view_id -> image}.
    :param _global_instance_id: Counter to generate global instance IDs
        for Instance class.
    :param _instance_cluster_id: Counter to generate cluster IDs for
        InstanceCluster class.
    """

    visual_difference_threshold: float
    embedding_model: EmbeddingNet
    name: str = "ReactManager"
    include_z: bool = False
    _instance_clusters: Dict[int, InstanceCluster] = field(default_factory=dict)
    _map_views: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    _global_instance_id: int = 0
    _instance_cluster_id: int = 0

    def pretty_print(self) -> str:
        """Return a string representation of the ReactManager.

        This method provides a detailed string representation of the
        ReactManager, including its clusters.

        :return: A string representation of the ReactManager.
        """
        updater_str = f"\n🌞 {self.name}:\n" + f"- Clusters:\n"
        for cluster in self._instance_clusters.values():
            updater_str += cluster.pretty_print()
        return updater_str

    def get_instance_clusters(self):
        """Get the instance clusters managed by ReactManager.

        :return: The instance clusters.
        """
        return self._instance_clusters

    def assign_instance_clusters(self, instance_clusters: Dict[int, InstanceCluster]):
        """Assign instance clusters to ReactManager.

        :param instance_clusters: A dictionary of instance clusters.
        """
        self._instance_clusters = deepcopy(instance_clusters)

    def assign_instance_id(self) -> int:
        """Assign a new global instance ID.

        This method returns a new ID and increments the global instance
        ID counter.

        :return: A new global instance ID.
        """
        id = self._global_instance_id
        self._global_instance_id += 1
        return id

    def assign_cluster_id(self) -> int:
        """Assign a new cluster ID.

        This method returns the new ID and increments the cluster ID
        counter.

        :return: A new cluster ID.
        """
        id = self._instance_cluster_id
        self._instance_cluster_id += 1
        return id

    def get_object_nodes_from_json_data(
        self,
        scan_id: int,
        instance_views_data: Dict,
        dsg_data: Dict,
    ) -> Dict[int, ObjectNode]:
        """Get object nodes from JSON data.

        This method processes the given JSON data and retrieves object
        nodes.

        :param scan_id: The scan ID.
        :param instance_views_data: Instance views data from
            instance_views.json.
        :param dsg_data: DSG data from dsg_with_mesh.json.
        :return: A dictionary of object nodes, mapped as a dictionary
            {global_instance_id : ObjectNode}.
        """
        object_nodes: Dict[int, ObjectNode] = {}
        num_features = self.embedding_model.num_features()
        scan_map_views = self._map_views[scan_id]
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
                    map_view_img=scan_map_views[map_view_id],
                    mask=mask,
                    mask_bg=True,
                    crop=True,
                    padding=10,
                )
                preprocessed_img = preprocess_image(view_img)
                embedding = self.embedding_model(preprocessed_img).detach().cpu()
                # view_embeddings += embedding
                view_embeddings[i, :] = embedding
            # view_embeddings /= len(all_masks_data)
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
            object_nodes[self.assign_instance_id()] = new_node
        for node in object_nodes.values():
            logger.debug(node.pretty_print())
        return object_nodes

    def merge_two_clusters(self, cluster_id: int, other_cluster_id: int):
        """Merge two instance clusters.

        This method merges two instance clusters into a single cluster.

        :param cluster_id: The ID of the first cluster.
        :param other_cluster_id: The ID of the second cluster.
        """
        assert (
            cluster_id in self._instance_clusters
        ), f"Cluster ID is not in {self.name}: {cluster_id}"
        assert (
            other_cluster_id in self._instance_clusters
        ), f"Cluster ID is not in {self.name}: {other_cluster_id}"
        cluster: InstanceCluster = self._instance_clusters.pop(cluster_id)
        other_cluster: InstanceCluster = self._instance_clusters.pop(other_cluster_id)
        new_cluster_id = min(cluster_id, other_cluster_id)
        new_instances = cluster.instances
        for inst in other_cluster.instances.values():
            new_instances.update({inst.global_id: inst})
        new_cluster = InstanceCluster(
            cluster_id=new_cluster_id, instances=new_instances
        )
        self._instance_clusters.update({new_cluster_id: new_cluster})

    def init_instance_clusters(self, scan_id: int, object_nodes: Dict[int, ObjectNode]):
        """Initialize instance clusters with object nodes.

        This method initializes instance clusters using the provided
        object nodes.

        :param scan_id: The scan ID.
        :param object_nodes: A dictionary of object nodes, mapped as a
            dictionary {global_instance_id : ObjectNode}.
        """
        for global_instance_id, node in object_nodes.items():
            cluster_id = self.assign_cluster_id()
            instance = Instance(
                global_id=global_instance_id, node_history={scan_id: node}
            )
            instance_cluster = InstanceCluster(
                cluster_id=cluster_id,
                instances={global_instance_id: instance},
            )
            self._instance_clusters.update({cluster_id: instance_cluster})

    def optimize_cluster(self):
        """Optimize the clustering of instances.

        This method iteratively merges clusters that match based on
        visual embeddings until no more matches are found.
        """
        while True:
            match_pair = None
            for cid, cluster in self._instance_clusters.items():
                if match_pair is not None:
                    break
                for other_cid, other_cluster in self._instance_clusters.items():
                    if (
                        cid != other_cid
                        and cluster.get_class_id() == other_cluster.get_class_id()
                    ):
                        is_match = cluster.is_match(
                            other_cluster=other_cluster,
                            visual_difference_threshold=self.visual_difference_threshold,
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

    def greedy_match(self, scan_id_old: int, scan_id_new: int):
        """Perform greedy matching of object nodes between two scans.

        This method matches object nodes between an old scan and a new
        scan by minimizing the traveled distance of objects, assuming
        all instance clusters only have one object node each.

        :param scan_id_old: The old scan ID.
        :param scan_id_new: The new scan ID.
        """
        # cluster_id -> node
        old_clusters: Dict[int, ObjectNode] = {}
        new_clusters: Dict[int, ObjectNode] = {}
        matched_ids = []
        for c in self._instance_clusters.values():
            for inst_id, instance in c.instances.items():
                if scan_id_new in instance.node_history:
                    node = instance.node_history[scan_id_new]
                    new_clusters.update({c.cluster_id: node})

                elif scan_id_old in instance.node_history:
                    node = instance.node_history[scan_id_old]
                    old_clusters.update({c.cluster_id: node})

        for new_cid, new_node in new_clusters.items():
            assert isinstance(
                new_node.embedding, Tensor
            ), f"Node {new_node.node_id} does not have embeddings"
            dists = []
            candidate_cluster_ids = []
            for old_cid, old_node in old_clusters.items():
                if (
                    old_cid not in matched_ids
                    and old_node.class_id == new_node.class_id
                ):
                    dists.append(np.linalg.norm(new_node.position - old_node.position))
                    candidate_cluster_ids.append(old_cid)
            sorted_dists = np.argsort(np.array(dists))
            for idx in sorted_dists:
                old_cid = candidate_cluster_ids[idx]
                emb_dist = torch.nn.functional.pairwise_distance(
                    self._instance_clusters[old_cid].get_embedding(),
                    self._instance_clusters[new_cid].get_embedding(),
                )
                if emb_dist < self.visual_difference_threshold:
                    matched_ids.append(old_cid)
                    self.merge_two_clusters(
                        cluster_id=old_cid, other_cluster_id=new_cid
                    )
                    break
        self.update_position_histories(scan_id_old=scan_id_old, scan_id_new=scan_id_new)

    def update_instance_entry(
        self, cluster_id: int, old_inst_id: int, new_inst_id: int
    ):
        """Update an instance entry by merging two instances within a cluster.

        This method merges two instances in a specific cluster and
        assigns the minimum ID to the merged instance.

        :param cluster_id: The ID of the cluster containing the
            instances.
        :param old_inst_id: The instance ID of the first instance.
        :param new_inst_id: The instance ID of the second instance.
        """
        self._instance_clusters[cluster_id].merge_two_instances(
            inst_id=old_inst_id,
            other_inst_id=new_inst_id,
            assign_inst_id=min(old_inst_id, new_inst_id),
        )

    def update_position_histories(self, scan_id_old: int, scan_id_new: int):
        """Update position histories between two scans for all clusters.

        This method updates the position histories between two scans for
        all clusters, matching instances from the old scan to instances
        from the new scan.

        :param scan_id_old: The old scan ID.
        :param scan_id_new: The new scan ID.
        """
        for inst_cluster_id, inst_cluster in self._instance_clusters.items():
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
        """Get all object nodes in a specific scan.

        This method retrieves all object nodes present in a specific
        scan across all instance clusters.

        :param scan_id: The scan ID to retrieve nodes from.
        :return: A dictionary of nodes with their instance IDs, mapped
            as a dictionary {global_instance_id : ObjectNode}.
        """
        nodes = {}
        for cluster in self._instance_clusters.values():
            for inst_id, instance in cluster.instances.items():
                if scan_id in instance.node_history:
                    node = instance.node_history[scan_id]
                    nodes.update({inst_id: node})
        return nodes

    def report_match_results(
        self, old_scan_id: int = 0, new_scan_id: int = 1
    ) -> MatchResults:
        """Report matching results between two scans.

        This method reports the matching results between two scans,
        including travel distance, matches, absent nodes, and new nodes.

        :param old_scan_id: The old scan ID (default: 0).
        :param new_scan_id: The new scan ID (default: 1).
        :return: An object of MatchResults containing the matching
            details.
        """
        results = MatchResults(old_scan_id=old_scan_id, new_scan_id=new_scan_id)
        assert old_scan_id < new_scan_id
        for iset in self._instance_clusters.values():
            for ph in iset.get_cluster_position_history(
                scan_ids=[old_scan_id, new_scan_id]
            ).values():
                if old_scan_id in ph.keys() and new_scan_id in ph.keys():
                    results.travel_distance += np.linalg.norm(
                        ph[old_scan_id] - ph[new_scan_id]
                    ).astype(float)
            for nh in iset.get_node_history().values():
                if old_scan_id in nh and new_scan_id in nh:
                    results.matches.append(
                        (nh[old_scan_id].node_id, nh[new_scan_id].node_id)
                    )
                elif old_scan_id in nh and new_scan_id not in nh:
                    results.absent.append(nh[old_scan_id].node_id)
                elif old_scan_id not in nh and new_scan_id in nh:
                    results.new.append(nh[new_scan_id].node_id)
        return results

    def process_dsg(self, dsg_path: str, scan_id: int, optimize_cluster: bool = True):
        """Process DSG data from json files to initialize and optimize instance
        clusters.

        This method processes DSG data from dsg_with_mesh.json,
        instance_views.json and map_views.json to initialize instance
        clusters, and optionally optimize them.

        :param dsg_path: The path to the DSG data files.
        :param scan_id: The scan ID for the current data.
        :param optimize_cluster: Whether to optimize the clusters after
            initialization (default: True).
        """
        instance_views_data, map_views_data, dsg_data = get_dsg_data(dsg_path)
        self._map_views.update({scan_id: register_map_views(map_views_data)})
        object_nodes = self.get_object_nodes_from_json_data(
            scan_id=scan_id,
            instance_views_data=instance_views_data,
            dsg_data=dsg_data,
        )
        self.init_instance_clusters(scan_id=scan_id, object_nodes=object_nodes)
        if optimize_cluster:
            self.optimize_cluster()
