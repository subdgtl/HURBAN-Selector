use std::collections::HashSet;

use crate::bounding_box::BoundingBox;
use crate::interpreter::{LogMessage, MeshArrayValue};
use crate::mesh::{analysis, Mesh, UnorientedEdge};

pub fn report_mesh_analysis(mesh: &Mesh) -> Vec<LogMessage> {
    let vertex_count = mesh.vertices().len();
    let normal_count = mesh.normals().len();
    let face_count = mesh.faces().len();
    let edges: HashSet<UnorientedEdge> = mesh.unoriented_edges_iter().collect();
    let edge_count = edges.len();

    let bbox = mesh.bounding_box();
    let bbox_center = bbox.center();
    let bbox_dimensions = bbox.maximum_point() - bbox.minimum_point().coords;
    let bbox_diagonal = bbox.diagonal().norm();
    let has_no_orphan_normals = mesh.has_no_orphan_normals();
    let has_no_orphan_vertices = mesh.has_no_orphan_vertices();
    let is_triangulated = mesh.is_triangulated();

    let oriented_edges: Vec<_> = mesh.oriented_edges_iter().collect();
    let edge_sharing_map = analysis::edge_sharing(&oriented_edges);
    let is_watertight = analysis::is_mesh_watertight(&edge_sharing_map);
    let is_manifold = analysis::is_mesh_manifold(&edge_sharing_map);
    let is_orientable = analysis::is_mesh_orientable(&edge_sharing_map);

    let mut report: Vec<LogMessage> = Vec::new();

    report.push(if is_triangulated {
        LogMessage::info("Triangulated mesh properties:")
    } else {
        LogMessage::warn("Non-triangulated mesh properties:")
    });

    report.push(LogMessage::info(format!(
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

    report.push(if has_no_orphan_vertices && has_no_orphan_normals {
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

    report.push(if is_watertight && is_manifold && is_orientable {
        LogMessage::info(quality_report)
    } else {
        LogMessage::warn(quality_report)
    });

    if is_watertight {
        let genus = analysis::triangulated_mesh_genus(vertex_count, edge_count, face_count);
        report.push(LogMessage::info(format!(
            "Genus {} (number of topological holes)",
            genus
        )));
    } else {
        let edge_loops = analysis::border_edge_loops(&edge_sharing_map);
        let edge_loop_count = edge_loops.len();
        report.push(LogMessage::info(format!(
            "Has {} naked border {}",
            edge_loop_count,
            if edge_loop_count == 1 {
                "loop"
            } else {
                "loops"
            }
        )));
    }

    if !is_manifold {
        let non_manifold: Vec<_> = analysis::non_manifold_edges(&edge_sharing_map).collect();
        let non_manifold_count = non_manifold.len();
        report.push(LogMessage::warn(format!(
            "Has {} non-manifold edges",
            non_manifold_count
        )));
    }

    report.push(LogMessage::info("Bounding box properties:"));
    report.push(LogMessage::info(format!(
        "Center = [{:.2}, {:.2}, {:.2}]",
        bbox_center.x, bbox_center.y, bbox_center.y,
    )));
    report.push(LogMessage::info(format!(
        "Dimensions = [{:.2}, {:.2}, {:.2}]",
        bbox_dimensions.x, bbox_dimensions.y, bbox_dimensions.y,
    )));
    report.push(LogMessage::info(format!(
        "Diagonal length = {:.2}",
        bbox_diagonal
    )));

    report
}

pub fn report_group_analysis(group: &MeshArrayValue) -> Vec<LogMessage> {
    let group_len = group.len();
    let meshes = group.iter_refcounted();
    let bboxes: Vec<_> = meshes.map(|mesh| mesh.bounding_box()).collect();
    let union_bbox = BoundingBox::union(bboxes).expect("No input bounding boxes");

    let mut report: Vec<LogMessage> = Vec::new();

    report.push(LogMessage::info(format!(
        "The group contains {} {}",
        group_len,
        if group_len == 1 { "mesh" } else { "meshes" }
    )));

    let bbox_center = union_bbox.center();
    let bbox_dimensions = union_bbox.maximum_point() - union_bbox.minimum_point().coords;
    let bbox_diagonal = union_bbox.diagonal().norm();

    report.push(LogMessage::info("Union bounding box properties:"));
    report.push(LogMessage::info(format!(
        "center = [{:.2}, {:.2}, {:.2}]",
        bbox_center.x, bbox_center.y, bbox_center.y,
    )));
    report.push(LogMessage::info(format!(
        "dimensions = [{:.2}, {:.2}, {:.2}]",
        bbox_dimensions.x, bbox_dimensions.y, bbox_dimensions.y,
    )));
    report.push(LogMessage::info(format!(
        "diagonal length = {:.2}",
        bbox_diagonal
    )));

    report
}
