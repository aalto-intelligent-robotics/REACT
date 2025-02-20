from dataclasses import dataclass
import open3d as o3d
from typing import List
import numpy as np

from react.utils.logger import getLogger

logger = getLogger(name=__name__, log_file="bounding_box.log")


@dataclass
class BoundingBox:
    """
    Bounding box class for object nodes

    Attributes: 
        min_bounds: min x, y, z (in m)
        max_bounds: max x, y, z (in m)
    """
    min_bounds: np.ndarray
    max_bounds: np.ndarray

    def create_o3d_aabb(self, color: List) -> o3d.geometry.AxisAlignedBoundingBox:
        bbox_o3d = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=self.min_bounds, max_bound=self.max_bounds
        )
        bbox_o3d.color = color
        return bbox_o3d

    def get_dims(self) -> np.ndarray:
        return self.max_bounds - self.min_bounds

    def get_center(self) -> np.ndarray:
        return self.max_bounds - (self.get_dims() / 2)
