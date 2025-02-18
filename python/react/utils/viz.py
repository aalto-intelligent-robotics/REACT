from copy import deepcopy
import open3d as o3d
from typing import List, Dict
import numpy as np

from hydra_seg_ros.utils.labels import COCO_COLORS
from react.core.object_node import ObjectNode
from react.core.instance_cluster import InstanceCluster
from react.core.map_updater import MapUpdater


def get_class_color(class_id: int, normalize: bool = True) -> List:
    color = COCO_COLORS[class_id]
    return [float(c) / 255 for c in color] if normalize else list(color)


def extract_submesh(
    mesh: o3d.geometry.TriangleMesh, node: ObjectNode
) -> o3d.geometry.TriangleMesh:
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
    # Create the box mesh
    box = o3d.geometry.TriangleMesh.create_box(dim, dim, dim)
    # Translate the box to the specified center
    translation = center - np.array([dim / 2, dim / 2, dim / 2])
    box.translate(translation)
    box.paint_uniform_color(color)
    return box


def draw_text_mesh(text: str, position: np.ndarray) -> o3d.geometry.TriangleMesh:
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
    map_updater: MapUpdater,
    node_label_z: float = 3,
    set_label_z: float = 5,
    dsg_offset_z: float = 0,
    include_scene_mesh: bool = True,
    include_instance_mesh: bool = True,
    draw_set: bool = True,
    draw_text: bool = False,
) -> List:
    object_nodes = map_updater.get_nodes_in_scan(scan_id=scan_id)
    instance_clusters = map_updater._instance_clusters
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
    map_updater: MapUpdater,
    old_dsg_offset_z: float = 0,
    new_dsg_offset_z: float = -5,
    include_instance_mesh: bool = True,
    include_scene_mesh: bool = True,
    draw_text=False,
) -> List:
    object_nodes = map_updater.get_nodes_in_scan(scan_id=scan_id_new)
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

    for inst_cluster in map_updater._instance_clusters.values():
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
