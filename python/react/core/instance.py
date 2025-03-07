import numpy as np
import logging
from typing import Dict, List
from dataclasses import dataclass

from react.core.object_node import ObjectNode
from react.utils.logger import getLogger

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="instance.log",
)


@dataclass
class Instance:
    """Global information of an instance through multiple scans.

    This class represents global information about an instance across
    multiple scans. It stores the instance's global ID and a history of
    nodes representing the instance in different scans.

    :param global_id: Global ID of the instance.
    :param node_history: A dictionary mapping scan IDs to ObjectNode
        instances representing the instance in different scans.
    """

    global_id: int
    node_history: Dict[int, ObjectNode]

    def pretty_print(self) -> str:
        """Return a string representation of the instance.

        This method provides a detailed string representation of the
        instance, including its global ID and position history.

        :return: A string representation of the instance.
        """
        instance_str = (
            f"\n⭐ Instance Info:\n"
            + f"- Global Instance ID: {self.global_id}\n"
            + f"- Position History: {self.get_position_history(scan_ids=[])}\n"
        )
        return instance_str

    def get_position_history(
        self, scan_ids: List[int] = [0, 1]
    ) -> Dict[int, np.ndarray]:
        """Return the position history between specified scans.

        This method retrieves the position history of the instance for
        the given scan IDs. If no scan IDs are provided, it returns the
        positions for all scans.

        :param scan_ids: The scan IDs to get history from. Default is
            [0, 1].
        :return: The history of the instance mapped as a dictionary
            {scan_id: position}.
        """
        ph = {}
        if scan_ids:
            for node in self.node_history.values():
                if node.scan_id in scan_ids:
                    ph.update({node.scan_id: node.position})
        else:
            for node in self.node_history.values():
                ph.update({node.scan_id: node.position})
        return ph

    def get_class_id(self) -> int:
        """Get the instance class label.

        This method returns the class label of the instance by accessing
        the class ID of the first node in the node history.

        :return: The class label of the instance.
        """
        node = next(iter(self.node_history.values()))
        return node.class_id

    def get_name(self) -> str:
        """Get the instance name.

        This method returns the name of the instance by accessing the
        name of the first node in the node history.

        :return: The name of the instance.
        """
        node = next(iter(self.node_history.values()))
        return node.name

    def empty(self) -> bool:
        """Check if the instance is empty.

        This method checks if the instance has no associated nodes in
        its history.

        :return: True if the instance is empty, False otherwise.
        """
        return len(self.node_history) == 0
