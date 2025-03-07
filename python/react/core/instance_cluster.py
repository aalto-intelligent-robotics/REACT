import logging
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from react.core.object_node import ObjectNode
import torch
from torch import Tensor

from react.utils.logger import getLogger
from react.core.instance import Instance
from react.matching.hungarian_algorithm import hungarian_algorithm

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="instance_sets.log",
)


@dataclass
class InstanceCluster:
    """Python interface for REACT Instance Cluster for managing Instance groups
    that represent identical objects.

    :param cluster_id: ID of the cluster.
    :param instances: A collection of instances mapped by their global
        instance ID.
    """

    cluster_id: int
    instances: Dict[int, Instance]

    def pretty_print(self) -> str:
        """Return a string representation of the instance cluster.

        This method provides a detailed string representation of the
        instance cluster, including its ID, class ID, name, embedding,
        and instances.

        :return: A string representation of the instance cluster.
        """
        node_str = (
            "\n🌍 Instance Cluster info:\n"
            + f"- Cluster ID: {self.cluster_id}\n"
            + f"- Class ID: {self.get_class_id()}\n"
            + f"- Name: {self.get_name()}\n"
            + f"- Embedding: {self.get_embedding()}\n"
            + f"- Instances:\n"
        )
        for instance in self.instances.values():
            node_str += instance.pretty_print()
        node_str += "\n"
        return node_str

    def get_class_id(self) -> int:
        """Get the class ID of the instance cluster.

        This method returns the class ID of the instance cluster by
        accessing the class ID of the first instance.

        :return: The class ID of the instance cluster.
        """
        instance = next(iter(self.instances.values()))
        return instance.get_class_id()

    def get_name(self) -> str:
        """Get the name of the instance cluster.

        This method returns the name of the instance cluster by
        accessing the name of the first instance.

        :return: The name of the instance cluster.
        """
        instance = next(iter(self.instances.values()))
        return instance.get_name()

    def get_local_node_id(self, scan_id: int, global_object_id: int) -> int:
        """Return the ID of an ObjectNode given a scan ID and the global object
        ID.

        :param scan_id: The scan ID of the scene.
        :param global_object_id: The global object ID registered in the
            object cluster.
        :return: The ObjectNode ID.
        """
        return self.instances[global_object_id].node_history[scan_id].node_id

    def get_embedding(self) -> Tensor:
        """Return the average embedding for the cluster.

        This method calculates the weighted average embedding for the
        cluster based on the number of images a node has.

        :return: The average embedding for the cluster.
        """
        embedding = None
        num_imgs = 0

        for global_node in self.instances.values():
            for node in global_node.node_history.values():
                node_num_imgs = len(node.instance_views)
                assert node.embedding is not None

                if embedding is None:
                    embedding = node.embedding * node_num_imgs
                else:
                    embedding += node.embedding * node_num_imgs

                num_imgs += node_num_imgs

        assert isinstance(embedding, Tensor)
        assert num_imgs > 0
        embedding /= num_imgs
        return embedding

    def get_cluster_position_history(
        self, scan_ids: List[int] = [0, 1]
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Get the position history of the nodes for specified scans in the
        cluster.

        :param scan_ids: Scan IDs to get history from.
        :return: Position histories of instances within the cluster
            mapped as {instance_id -> {scan_id -> pos}}.
        """
        ph = {}
        for instance_id, instance in self.instances.items():
            ph.update({instance_id: instance.get_position_history(scan_ids=scan_ids)})
        return ph

    def get_node_history(
        self, scan_ids: List[int] = [0, 1]
    ) -> Dict[int, Dict[int, ObjectNode]]:
        """Get the object node history for specified scans in the cluster.

        :param scan_ids: Scan IDs to get history from. Set to [] to
            select from all scans.
        :return: Object nodes histories of instances within the cluster
            mapped as {instance_id -> {scan_id -> object_node}}.
        """
        nh = {}
        for instance_id, instance in self.instances.items():
            for scan, node in instance.node_history.items():
                if len(scan_ids) == 0:
                    is_valid_scan = True
                else:
                    is_valid_scan = True if scan in scan_ids else False
                if is_valid_scan:
                    if instance_id in nh:
                        nh[instance_id].update({scan: node})
                    else:
                        nh[instance_id] = {scan: node}
        return nh

    def is_match(
        self, other_cluster: "InstanceCluster", visual_difference_threshold: float
    ) -> bool:
        """Compare with another InstanceCluster.

        This method compares the current instance cluster with another
        instance cluster to determine if they represent similar objects
        based on a visual difference threshold.

        :param other_cluster: The other instance cluster.
        :param visual_difference_threshold: The visual difference
            threshold.
        :return: True if the two clusters represent similar objects
            according to the visual difference threshold, False
            otherwise.
        """
        dist = torch.nn.functional.pairwise_distance(
            self.get_embedding(), other_cluster.get_embedding()
        )
        logger.debug(
            f"Dist cluster {self.get_name()} {self.cluster_id}"
            + f" - {other_cluster.get_name()} {other_cluster.cluster_id}"
            + f": {dist}"
        )
        if dist <= visual_difference_threshold:
            return True
        return False

    def match_position(
        self, scan_id_old: int, scan_id_new: int, include_z: bool = False
    ) -> Tuple[List, List]:
        """Perform instance matching by minimizing traveled distance of objects
        using the Hungarian algorithm.

        This method uses the Hungarian algorithm on instances that
        contain two ObjectNodes from the new and old scan IDs to perform
        instance matching by minimizing the traveled distance.

        :param scan_id_old: The old scan ID.
        :param scan_id_new: The new scan ID.
        :param include_z: Whether to include the z-coordinate in
            distance calculations. Default is False.
        :return: A tuple containing lists of instance IDs in the old
            scan and their matching correspondences in the new scan.
        """
        old_inst_positions = {}
        new_inst_positions = {}
        for inst_id, ph in self.get_cluster_position_history(
            scan_ids=[scan_id_old, scan_id_new]
        ).items():
            if scan_id_old in ph:
                old_inst_positions[inst_id] = ph[scan_id_old]
            if scan_id_new in ph:
                new_inst_positions[inst_id] = ph[scan_id_new]
        matching_old_ids, matching_new_ids = hungarian_algorithm(
            old_inst_positions=old_inst_positions,
            new_inst_positions=new_inst_positions,
            include_z=include_z,
        )
        return matching_old_ids, matching_new_ids

    def merge_two_instances(
        self, inst_id: int, other_inst_id: int, assign_inst_id: int
    ):
        """Merge two Instance objects.

        This method merges two Instance objects within the cluster and
        assigns the merged instance a new instance ID.

        :param inst_id: The instance ID of the first Instance object.
        :param other_inst_id: The instance ID of the second Instance
            object.
        :param assign_inst_id: The instance ID to assign to the merged
            Instance object.
        """
        assert (
            inst_id in self.instances
        ), f"Instance ID to be merged is not available: {inst_id}"
        assert (
            other_inst_id in self.instances
        ), f"Instance ID to b merged is not available: {other_inst_id}"
        inst: Instance = self.instances.pop(inst_id)
        other_inst: Instance = self.instances.pop(other_inst_id)
        new_node_history = inst.node_history
        for node in other_inst.node_history.values():
            new_node_history.update({node.scan_id: node})
        new_inst = Instance(global_id=assign_inst_id, node_history=new_node_history)
        self.instances.update({assign_inst_id: new_inst})

    def empty(self) -> bool:
        """Check if the cluster is empty.

        This method checks if there are no instances in the cluster.

        :return: True if the cluster is empty, False otherwise.
        """
        return len(self.instances) == 0
