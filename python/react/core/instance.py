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

    :param global_id: global id of the instance
    :param node_history: the nodes representing the instance in
        different scans {scan_id -> ObjectNode}
    """

    global_id: int
    node_history: Dict[int, ObjectNode]

    def __str__(self) -> str:
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

        :param scan_ids: the scan ids to get history from
        :return: The history of the instance mapped as {scan_id ->
            position}
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

        :return: The instance class label
        """
        node = next(iter(self.node_history.values()))
        return node.class_id

    def get_name(self) -> str:
        """Get the instance name.

        :return: The instance name
        """
        node = next(iter(self.node_history.values()))
        return node.name

    def empty(self) -> bool:
        """Checks if an instance is empty.

        :return: True if the instance is empty, False if it isn't
        """
        return len(self.node_history) == 0
