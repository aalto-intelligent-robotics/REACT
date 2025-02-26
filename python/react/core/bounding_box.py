from dataclasses import dataclass
import open3d as o3d
from typing import List
import numpy as np
import logging

from react.utils.logger import getLogger

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="bounding_box.log",
)


@dataclass
class BoundingBox:
    """Bounding box class for object nodes.

    :param min_bounds: min x, y, z (in m)
    :param max_bounds: max x, y, z (in m)
    """

    min_bounds: np.ndarray
    max_bounds: np.ndarray

    def create_o3d_aabb(self, color: List) -> o3d.geometry.AxisAlignedBoundingBox:
        """Create an Open3d AxisAlignedBoundingBox for visualization.

        :param color: the color of the bbox
        :return: an Open3d AxisAlignedBoundingBox
        """
        bbox_o3d = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=self.min_bounds, max_bound=self.max_bounds
        )
        bbox_o3d.color = color
        return bbox_o3d

    def get_dims(self) -> np.ndarray:
        """Returns the bbox dimensions.

        :return: the bbox dimensions (length, width, height)
        """
        return self.max_bounds - self.min_bounds

    def get_center(self) -> np.ndarray:
        """Returns the center of the bounding box (in min and max bounds'
        coordinates)

        :return: the center of the bounding box (in min and max bounds'
            coordinates)
        """
        return self.max_bounds - (self.get_dims() / 2)
