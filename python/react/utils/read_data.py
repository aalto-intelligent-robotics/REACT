import numpy as np
from typing import Dict, Tuple, List
import cv2
import json

from react.core.bounding_box import BoundingBox


def get_bbox(dimensions: List, position: List) -> BoundingBox:
    d, w, h = dimensions
    xc, yc, zc = position
    xmin, xmax = xc - d / 2, xc + d / 2
    ymin, ymax = yc - w / 2, yc + w / 2
    zmin, zmax = zc - h / 2, zc + h / 2
    return BoundingBox(
        min_bounds=np.array([xmin, ymin, zmin]), max_bounds=np.array([xmax, ymax, zmax])
    )


def get_node_attrs(dsg_data, node_id) -> Dict:
    for node_data in dsg_data["nodes"]:
        if node_data["id"] == node_id:
            return node_data["attributes"]
    return {}


def register_map_views(map_views_data) -> Dict[int, np.ndarray]:
    map_views = {}
    for view_data in map_views_data:
        map_view_file = view_data["file"]
        map_view_id = view_data["map_view_id"]
        map_views[map_view_id] = cv2.imread(map_view_file)
    return map_views


def get_dsg_data(dsg_path) -> Tuple[Dict, Dict, Dict]:
    with open(f"{dsg_path}/instance_views/instance_views.json") as f:
        instance_views_data = json.load(f)
    with open(f"{dsg_path}/map_views/map_views.json") as f:
        map_views_data = json.load(f)
    with open(f"{dsg_path}/backend/dsg_with_mesh.json") as f:
        dsg_data = json.load(f)
    return instance_views_data, map_views_data, dsg_data
