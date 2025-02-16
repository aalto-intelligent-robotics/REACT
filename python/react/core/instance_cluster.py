import logging
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
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
    """
    Python interface for REACT Instance Cluster

    Attributes:
        cluster_id: Id of the cluster
        instances: a collection of instances {global_instance_id -> Instance}
    """

    cluster_id: int
    instances: Dict[int, Instance]

    def __str__(self) -> str:
        node_str = (
            "\n🌍 Instance Cluster info:\n"
            + f"- Cluster ID: {self.cluster_id}\n"
            + f"- Class ID: {self.get_class_id()}\n"
            + f"- Name: {self.get_name()}\n"
            + f"- Embedding: {self.get_embedding()}\n"
            + f"- Instances:\n"
        )
        for instance in self.instances.values():
            node_str += instance.__str__()
        node_str += "\n"
        return node_str

    def get_class_id(self) -> int:
        instance = next(iter(self.instances.values()))
        return instance.get_class_id()

    def get_name(self) -> str:
        instance = next(iter(self.instances.values()))
        return instance.get_name()

    def get_local_node_id(self, scan_id: int, global_object_id: int) -> int:
        """
        Return the id of an ObjectNode given a scan id and the global object id

        Args:
            scan_id: the scan id of the scene
            global_object_id: the global object id registered in the object cluster

        Returns:
            the ObjectNode id
        """
        return self.instances[global_object_id].node_history[scan_id].node_id

    def get_embedding(self) -> Tensor:
        """
        Return the average embedding for the cluster. Note that this takes the weighted
        average depending on how many images a node has

        Returns:
            The average emdbedding for the cluster
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
        """
        Get the position history of the nodes in the cluster

        Args:
            scan_ids: scan ids to get history from

        Returns:
            position histories of instances within cluster mapped as
            {instance_id -> {scan_id -> pos}}
        """
        ph = {}
        for instance_id, instance in self.instances.items():
            ph.update({instance_id: instance.get_position_history(scan_ids=scan_ids)})
        return ph

    def is_match(self, other_cluster: "InstanceCluster", match_threshold: float):
        dist = torch.nn.functional.pairwise_distance(
            self.get_embedding(), other_cluster.get_embedding()
        )
        logger.debug(
            f"Dist cluster {self.get_name()} {self.cluster_id} - {self.get_name()} {self.cluster_id}: {dist}"
        )
        if dist <= match_threshold:
            return True
        return False

    def match_position(
        self, scan_id_old: int, scan_id_new: int, include_z: bool = False
    ):
        # Position Histories: {instance_id -> {scan_id -> pos}}
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

    def merge_two_instances(self, inst_id: int, other_inst_id: int):
        assert (
            inst_id in self.instances
        ), f"Instance ID to be merged is not available: {inst_id}"
        assert (
            other_inst_id in self.instances
        ), f"Instance ID to b merged is not available: {other_inst_id}"
        inst: Instance = self.instances.pop(inst_id)
        other_inst: Instance = self.instances.pop(other_inst_id)
        new_inst_id = min(inst_id, other_inst_id)
        new_node_history = inst.node_history
        for node in other_inst.node_history.values():
            new_node_history.update({node.scan_id: node})
        new_inst = Instance(global_id=new_inst_id, node_history=new_node_history)
        self.instances.update({new_inst_id: new_inst})

    def empty(self) -> bool:
        return len(self.instances) == 0
