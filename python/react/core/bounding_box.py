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

    This class represents a bounding box defined by its minimum and
    maximum bounds in 3D space. It provides methods to create an Open3D
    AxisAlignedBoundingBox for visualization, as well as to get the
    dimensions and the center of the bounding box.

    :param min_bounds: min x, y, z (in m)
    :param max_bounds: max x, y, z (in m)
    """

    min_bounds: np.ndarray
    max_bounds: np.ndarray

    def create_o3d_aabb(
        self, color: List[float]
    ) -> o3d.geometry.AxisAlignedBoundingBox:
        """Create an Open3D AxisAlignedBoundingBox for visualization.

        This method creates and returns an Open3D AxisAlignedBoundingBox
        object with the specified color for visualization purposes.

        :param color: The RGB color of the bounding box normalized to
            [0,1].
        :return: The Open3D AxisAlignedBoundingBox object.
        """
        bbox_o3d = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=self.min_bounds, max_bound=self.max_bounds
        )
        bbox_o3d.color = color
        return bbox_o3d

    def get_dims(self) -> np.ndarray:
        """Get the dimensions of the bounding box.

        This method returns the dimensions (length, width, height) of
        the bounding box by calculating the differences between the
        maximum and minimum bounds.

        :return: The dimensions of the bounding box (length, width,
            height).
        """
        return self.max_bounds - self.min_bounds

    def get_center(self) -> np.ndarray:
        """Get the center of the bounding box.

        This method returns the center coordinates of the bounding box
        by averaging the minimum and maximum bounds.

        :return: The center coordinates of the bounding box.
        """
        return self.max_bounds - (self.get_dims() / 2)
