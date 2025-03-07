from copy import deepcopy
import open3d as o3d
from typing import List, Dict
import numpy as np

from hydra_seg_ros.utils.labels import COCO_COLORS
from react.core.object_node import ObjectNode
from react.core.instance_cluster import InstanceCluster
from react.core.react_manager import ReactManager


def get_class_color(class_id: int, normalize: bool = True) -> List:
    """Get the color corresponding to a class ID from COCO_COLORS.

    :param class_id: The class ID for which to retrieve the color.
    :param normalize: Whether to normalize the color values to [0, 1].
        Defaults to True.
    :return: A list containing the color values.
    """
    color = COCO_COLORS[class_id]
    return [float(c) / 255 for c in color] if normalize else list(color)


def extract_submesh(
    mesh: o3d.geometry.TriangleMesh, node: ObjectNode
) -> o3d.geometry.TriangleMesh:
    """Extract a submesh from the given mesh based on the connections in the
    node.

    :param mesh: The original mesh to extract the submesh from.
    :param node: The ObjectNode containing the mesh connections.
    :return: The extracted submesh as an o3d.geometry.TriangleMesh
        object.
    """
    vertex_indices_set = node.mesh_connections
    # Get all triangles and check which ones have vertices in the vertex_indices
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    # Mask for triangles to keep
    triangle_mask = [
        all(v in vertex_indices_set for v in triangle) for triangle in triangles
    ]
    # Extract the triangles and vertices for the submesh
    selected_triangles = triangles[triangle_mask]
    unique_vertex_indices = np.unique(selected_triangles)
    mapping = {old: new for new, old in enumerate(unique_vertex_indices)}
    new_triangles = np.array(
        [[mapping[v] for v in triangle] for triangle in selected_triangles]
    )
    new_vertices = vertices[unique_vertex_indices]
    # Create the submesh
    submesh = o3d.geometry.TriangleMesh()
    submesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    submesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    color = get_class_color(class_id=node.class_id)
    submesh.paint_uniform_color(color)  # Paint class_color
    return submesh


def draw_cube(
    center: np.ndarray, color: List = [0.5, 0.5, 0.5], dim: float = 0.1
) -> o3d.geometry.TriangleMesh:
    """Draw a cube at a given center with specified dimensions and color.

    :param center: The center position of the cube.
    :param color: The color of the cube. Defaults to [0.5, 0.5, 0.5].
    :param dim: The dimension of the cube. Defaults to 0.1.
    :return: An o3d.geometry.TriangleMesh object representing the cube.
    """
    # Create the box mesh
    box = o3d.geometry.TriangleMesh.create_box(dim, dim, dim)
    # Translate the box to the specified center
    translation = center - np.array([dim / 2, dim / 2, dim / 2])
    box.translate(translation)
    box.paint_uniform_color(color)
    return box


def draw_text_mesh(text: str, position: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Create and draw a 3D text mesh at a specified position.

    :param text: The text to display in the mesh.
    :param position: The position where the text should be placed.
    :return: An o3d.geometry.TriangleMesh object representing the text.
    """
    # Create a 3D text mesh
    text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.5)
    text_mesh = text_mesh.to_legacy()
    text_mesh.scale(0.005, center=[0.0, 0.0, 0.0])
    text_mesh.rotate(
        text_mesh.get_rotation_matrix_from_xyz((np.pi / 3, 0, 0)), center=(0, 0, 0)
    )
    text_mesh.translate(position + np.array([0, 0, 0.1]))
    text_mesh.paint_uniform_color([0.0, 0.0, 0.0])
    return text_mesh


def draw_node_geometry(
    node: ObjectNode,
    mesh: o3d.geometry.TriangleMesh,
    node_label_z: float = 3,
    dsg_offset_z: float = 0,
    draw_bbox: bool = True,
    draw_mesh: bool = True,
    draw_lines: bool = True,
    draw_text: bool = True,
) -> Dict:
    """Draw the geometry for a specific node, including bounding box, lines,
    and text.

    :param node: The ObjectNode to draw.
    :param mesh: The mesh associated with the node.
    :param node_label_z: The z-axis position for the node label.
        Defaults to 3.
    :param dsg_offset_z: The z-axis offset for the DSG visualization.
        Defaults to 0.
    :param draw_bbox: Whether to draw the bounding box. Defaults to
        True.
    :param draw_mesh: Whether to draw the submesh. Defaults to True.
    :param draw_lines: Whether to draw the lines. Defaults to True.
    :param draw_text: Whether to draw the text label. Defaults to True.
    :return: A dictionary of the drawn geometries.
    """
    node_color = get_class_color(class_id=node.class_id)
    geometries = {}
    if draw_bbox:
        bbox = node.bbox.create_o3d_aabb(color=node_color)
        geometries["bbox"] = bbox
    if draw_lines:
        line_start = node.bbox.get_center()
        line_end = np.array([line_start[0], line_start[1], node_label_z])

        # Create a 3D text mesh
        if draw_text:
            text_mesh = draw_text_mesh(
                text=f"{node.name} {node.node_id}",
                position=line_end + np.array([0, 0, 0.1]),
            )
            geometries["text_mesh"] = text_mesh

        points = [line_start, line_end]
        lines = [[0, 1]]
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        geometries["line"] = line
        node_point = draw_cube(line_end, node_color)
        geometries["node_point"] = node_point
    for g in geometries.values():
        g.translate(np.array([0, 0, dsg_offset_z]))
    if draw_mesh:
        instance_mesh = extract_submesh(mesh=mesh, node=node)
        geometries["instance_mesh"] = instance_mesh
    return geometries


def draw_cluster_geometry(
    scan_id: int,
    node_geometries: Dict[int, Dict],
    instance_cluster: InstanceCluster,
    set_label_z: float = 5,
) -> Dict:
    """Draw the geometry for a cluster of instances, including lines and text
    labels.

    :param scan_id: The current scan ID.
    :param node_geometries: A dictionary of node geometries.
    :param instance_cluster: The InstanceCluster to draw.
    :param set_label_z: The z-axis position for the cluster label.
        Defaults to 5.
    :return: A dictionary of the drawn geometries.
    """
    color = get_class_color(instance_cluster.get_class_id())
    nodes_pos = []
    set_geometries = {}
    inst_ids_in_dsg = []
    for inst_id, pos in instance_cluster.get_cluster_position_history(
        scan_ids=[scan_id]
    ).items():
        if scan_id in pos.keys():
            nodes_pos.append(pos[scan_id])
            inst_ids_in_dsg.append(inst_id)
    if len(nodes_pos) > 0:
        set_center = np.mean(nodes_pos, axis=0)
        points = [set_center]
        lines = []
        set_center[2] = set_label_z
        text_mesh = draw_text_mesh(
            text=f"Set {instance_cluster.get_name()} {instance_cluster.cluster_id}",
            position=set_center,
        )
        for i, inst_id in enumerate(inst_ids_in_dsg):
            node_label_center = node_geometries[inst_id]["node_point"].get_center()
            points.append(node_label_center)
            lines.append([0, i + 1])
        lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        set_point = draw_cube(center=set_center, color=color)
        set_geometries = {
            "lines": lines,
            "set_point": set_point,
            "text_mesh": text_mesh,
        }
    return set_geometries


def draw_base_dsg(
    scan_id: int,
    mesh: o3d.geometry.TriangleMesh,
    react_manager: ReactManager,
    node_label_z: float = 3,
    set_label_z: float = 5,
    dsg_offset_z: float = 0,
    include_scene_mesh: bool = True,
    include_instance_mesh: bool = True,
    draw_set: bool = True,
    draw_text: bool = False,
) -> List:
    """Draw the base DSG visualization, including nodes, clusters, and
    optionally text labels.

    :param scan_id: The current scan ID.
    :param mesh: The mesh to include in the DSG visualization.
    :param react_manager: The ReactManager instance for updating the
        map.
    :param node_label_z: The z-axis position for the node labels.
        Defaults to 3.
    :param set_label_z: The z-axis position for the cluster labels.
        Defaults to 5.
    :param dsg_offset_z: The z-axis offset for the DSG visualization.
        Defaults to 0.
    :param include_scene_mesh: Whether to include the scene mesh or not.
        Defaults to True.
    :param include_instance_mesh: Whether to include the instance meshes
        or not. Defaults to True.
    :param draw_set: Whether to draw the scene Instance clusters or not.
        Defaults to True.
    :param draw_text: Whether to draw the text mesh or not. Defaults to
        True.
    :return: A collection of Open3D objects for visualization
    """
    object_nodes = react_manager.get_nodes_in_scan(scan_id=scan_id)
    instance_clusters = react_manager._instance_clusters
    mesh.translate(np.array([0, 0, dsg_offset_z]))
    dsg_viz = []
    if include_scene_mesh:
        dsg_viz.append(mesh)
    node_geometries = {}
    for id, node in object_nodes.items():
        node_viz = draw_node_geometry(
            node=node,
            mesh=mesh,
            node_label_z=node_label_z,
            dsg_offset_z=dsg_offset_z,
            draw_mesh=include_instance_mesh,
            draw_text=draw_text,
        )
        node_geometries[id] = node_viz
        dsg_viz += list(node_viz.values())
    if draw_set:
        for inst_cluster in instance_clusters.values():
            set_geometry = draw_cluster_geometry(
                scan_id=scan_id,
                node_geometries=node_geometries,
                instance_cluster=inst_cluster,
                set_label_z=set_label_z,
            )
            dsg_viz += list(set_geometry.values())

    return dsg_viz


def draw_matching_dsg(
    scan_id_old: int,
    scan_id_new: int,
    mesh: o3d.geometry.TriangleMesh,
    react_manager: ReactManager,
    old_dsg_offset_z: float = 0,
    new_dsg_offset_z: float = -5,
    include_instance_mesh: bool = True,
    include_scene_mesh: bool = True,
    draw_text=False,
) -> List:
    """Draw the matching DSG visualization, including nodes, instance matches
    and optionally text labels.

    :param scan_id_old: The reference scan ID.
    :param scan_id_new: The current scan ID.
    :param mesh: The mesh to include in the DSG visualization.
    :param react_manager: The ReactManager instance for updating the
        map.
    :param old_dsg_offset_z: The z-axis position for the nodes from the
        old scan. Defaults to 0.
    :param new_dsg_offset_z: The z-axis position for the nodes from the
        new scan. Defaults to -5.
    :param include_instance_mesh: Whether to include the instance meshes
        or not. Defaults to True.
    :param include_scene_mesh: Whether to include the scene mesh or not.
        Defaults to True.
    :return: A collection of Open3D objects for visualization
    """
    object_nodes = react_manager.get_nodes_in_scan(scan_id=scan_id_new)
    mesh.translate(np.array([0, 0, new_dsg_offset_z]))
    dsg_viz = []
    if include_scene_mesh:
        dsg_viz.append(mesh)
    node_geometries = {}
    for id, node in object_nodes.items():
        node_viz = draw_node_geometry(
            node=node,
            mesh=mesh,
            dsg_offset_z=new_dsg_offset_z,
            draw_lines=False,
            draw_mesh=include_instance_mesh,
            draw_text=draw_text,
        )
        node_geometries[id] = node_viz
        dsg_viz += list(node_viz.values())

    for inst_cluster in react_manager._instance_clusters.values():
        for inst_id, ph in inst_cluster.get_cluster_position_history(
            scan_ids=[scan_id_old, scan_id_new]
        ).items():
            if scan_id_new in ph.keys() and scan_id_old in ph.keys():
                old_pos = deepcopy(ph[scan_id_old])
                new_pos = deepcopy(ph[scan_id_new])
                old_pos[2] += old_dsg_offset_z
                new_pos[2] += new_dsg_offset_z
                points = [old_pos, new_pos]
                lines = [[0, 1]]
                lineset = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                dsg_viz.append(lineset)
    return dsg_viz
