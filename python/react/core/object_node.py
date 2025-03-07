from dataclasses import dataclass, field
import logging
from typing import Dict, Set, Union
import numpy as np
import torch

from react.core.bounding_box import BoundingBox
from react.utils.logger import getLogger

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="object_nodes.log",
)


@dataclass
class ObjectNode:
    """Python interface for Object Nodes in Hydra DSG.

    :param scan_id: id of the scan the node is in
    :param node_id: original node id in Hydra, only valid for one scan
    :param class_id: semantic class label
    :param name: node name, usually just semantic class
    :param position: xyz coordinates of centroid, registered by Hydra
    :param instance_views: library of images, stored as {map_view_id ->
        binary mask}
    :param bbox: bounding box of node (see react.core.bounding_box)
    :param embedding: visual embedding generated from embedding net (see
        react.net.embedding_net)
    :param mesh_connections: list of indices connected to current mesh
        (for visualization only)
    """

    scan_id: int
    node_id: int
    class_id: int
    name: str
    position: np.ndarray
    instance_views: Dict[int, np.ndarray]
    bbox: BoundingBox
    embedding: Union[torch.Tensor, None] = None
    mesh_connections: Set[int] = field(default_factory=set)

    def pretty_print(self) -> str:
        node_str = (
            "\n🌠 Node info:\n"
            + f"- Scan ID: {self.scan_id}\n"
            + f"- Node ID: {self.node_id}\n"
            + f"Class ID: {self.class_id}\n"
            + f"Name: {self.name}\n"
            + f"Position: {self.position}\n"
            + f"Embedding: {self.embedding}\n"
        )
        return node_str
