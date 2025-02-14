import logging
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from torch import Tensor

from react.utils.logger import getLogger
from react.core.instance import Instance

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
        instances: a collection of instances {global_object_id -> Instance}
    """

    cluster_id: int
    instances: Dict[int, Instance]

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
                assert node.embedding

                if not embedding:
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
            ph.update(
                {instance_id: instance.get_position_history(scan_ids=scan_ids)}
            )
        return ph

    def empty(self) -> bool:
        return len(self.instances) == 0
