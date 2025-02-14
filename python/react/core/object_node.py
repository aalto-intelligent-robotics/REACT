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
    """
    Python interface for Object Nodes in Hydra DSG

    Attributes:
        scan_id: id of the scan the node is in
        node_id: original node id in Hydra, only valid for one scan
        class_id: semantic class label
        name: node name, usually just semantic class
        position: xyz coordinates of centroid, registered by Hydra
        instance_views: library of images, stored as {map_view_id -> binary mask}
        bbox: bounding box of node (see react.core.bounding_box)
        embedding: visual embedding generated from embedding net (see
            react.net.embedding_net)
        mesh_connections: list of indices connected to current mesh (for visualization
            only)
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

    def __str__(self) -> str:
        node_str = (
            "🌠 Node info:\n"
            + f"- Scan ID: {self.scan_id}\n"
            + f"- Node ID: {self.node_id}\n"
            + f"Class ID: {self.class_id}\n"
            + f"Name: {self.name}\n"
            + f"Position: {self.position}"
        )
        return node_str
