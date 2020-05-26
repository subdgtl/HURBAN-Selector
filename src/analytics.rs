use std::collections::HashSet;

use crate::bounding_box::BoundingBox;
use crate::interpreter::{LogMessage, MeshArrayValue};
use crate::mesh::{analysis, Mesh, UnorientedEdge};

pub fn report_bounding_box_analysis(mesh: &Mesh, log: &mut dyn FnMut(LogMessage)) {
    let bbox = mesh.bounding_box();
    let bbox_center = bbox.center();
    let bbox_dimensions = bbox.diagonal();
    let bbox_diagonal_length = bbox_dimensions.norm();

    log(LogMessage::info("Bounding box properties:"));
    log(LogMessage::info(format!(
        "Center = [{:.2}, {:.2}, {:.2}]",
        bbox_center.x, bbox_center.y, bbox_center.z,
    )));
    log(LogMessage::info(format!(
        "Dimensions = [{:.2}, {:.2}, {:.2}]",
        bbox_dimensions.x, bbox_dimensions.y, bbox_dimensions.z,
    )));
    log(LogMessage::info(format!(
        "Diagonal length = {:.2}",
        bbox_diagonal_length
    )));
}

pub fn report_mesh_analysis(mesh: &Mesh, log: &mut dyn FnMut(LogMessage)) {
    let vertex_count = mesh.vertices().len();
    let normal_count = mesh.normals().len();
    let face_count = mesh.faces().len();
    let edges: HashSet<UnorientedEdge> = mesh.unoriented_edges_iter().collect();
    let edge_count = edges.len();

    let has_no_orphan_normals = mesh.has_no_orphan_normals();
    let has_no_orphan_vertices = mesh.has_no_orphan_vertices();
    let is_triangulated = mesh.is_triangulated();

    let oriented_edges: Vec<_> = mesh.oriented_edges_iter().collect();
    let edge_sharing_map = analysis::edge_sharing(&oriented_edges);
    let is_watertight = analysis::is_mesh_watertight(&edge_sharing_map);
    let is_manifold = analysis::is_mesh_manifold(&edge_sharing_map);
    let is_orientable = analysis::is_mesh_orientable(&edge_sharing_map);

    log(if is_triangulated {
        LogMessage::info("Triangulated mesh properties:")
    } else {
        LogMessage::warn("Non-triangulated mesh properties:")
    });

    log(LogMessage::info(format!(
        "{} vertices, {} {}, {} {}, {} edges",
        vertex_count,
        normal_count,
        if normal_count == 1 {
            "normal"
        } else {
            "normals"
        },
        face_count,
        if face_count == 1 { "face" } else { "faces" },
        edge_count,
    )));

    let orphan_report = format!(
        "{} orphan vertices, {} orphan normals",
        if has_no_orphan_vertices { "No" } else { "Has" },
        if has_no_orphan_normals { "no" } else { "has" }
    );

    log(if has_no_orphan_vertices && has_no_orphan_normals {
        LogMessage::info(orphan_report)
    } else {
        LogMessage::warn(orphan_report)
    });

    let quality_report = format!(
        "{} watertight, {}manifold, {} orientable",
        if is_watertight { "Is" } else { "Not" },
        if is_manifold { "is " } else { "non-" },
        if is_orientable { "is" } else { "not" },
    );

    log(if is_watertight && is_manifold && is_orientable {
        LogMessage::info(quality_report)
    } else {
        LogMessage::warn(quality_report)
    });

    if is_watertight {
        let genus = analysis::triangulated_mesh_genus(vertex_count, edge_count, face_count);
        log(LogMessage::info(format!(
            "Genus {} (number of topological holes)",
            genus
        )));
    } else {
        match analysis::border_edge_loops(&edge_sharing_map) {
            analysis::BorderEdgeLoopsResult::Found(edge_loops) => {
                let edge_loop_count = edge_loops.len();
                log(LogMessage::info(format!(
                    "Has {} valid naked border {}",
                    edge_loop_count,
                    if edge_loop_count == 1 {
                        "loop"
                    } else {
                        "loops"
                    }
                )));
            }
            analysis::BorderEdgeLoopsResult::FoundWithNondeterminism(edge_loops) => {
                let edge_loop_count = edge_loops.len();
                log(LogMessage::warn(format!(
                    "Has {} non-deterministic (potentially invalid) naked border {}",
                    edge_loop_count,
                    if edge_loop_count == 1 {
                        "loop"
                    } else {
                        "loops"
                    }
                )));
            }
            analysis::BorderEdgeLoopsResult::Watertight => {
                panic!("Should never come here for watertight meshes")
            }
        }
    }

    if !is_manifold {
        let non_manifold: Vec<_> = analysis::non_manifold_edges(&edge_sharing_map).collect();
        let non_manifold_count = non_manifold.len();
        log(LogMessage::warn(format!(
            "Has {} non-manifold edges",
            non_manifold_count
        )));
    }
}

pub fn report_group_analysis(group: &MeshArrayValue, log: &mut dyn FnMut(LogMessage)) {
    let group_len = group.len();
    let meshes = group.iter_refcounted();
    let bboxes = meshes.map(|mesh| mesh.bounding_box());
    let union_bbox = BoundingBox::union(bboxes).expect("No input bounding boxes");

    log(LogMessage::info(format!(
        "The group contains {} {}",
        group_len,
        if group_len == 1 { "mesh" } else { "meshes" }
    )));

    let bbox_center = union_bbox.center();
    let bbox_dimensions = union_bbox.diagonal();
    let bbox_diagonal_length = bbox_dimensions.norm();

    log(LogMessage::info("Union bounding box properties:"));
    log(LogMessage::info(format!(
        "Center = [{:.2}, {:.2}, {:.2}]",
        bbox_center.x, bbox_center.y, bbox_center.z,
    )));
    log(LogMessage::info(format!(
        "Dimensions = [{:.2}, {:.2}, {:.2}]",
        bbox_dimensions.x, bbox_dimensions.y, bbox_dimensions.z,
    )));
    log(LogMessage::info(format!(
        "Diagonal length = {:.2}",
        bbox_diagonal_length
    )));
}
