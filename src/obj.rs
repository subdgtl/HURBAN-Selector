use tobj;

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<Index>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
}

pub type Index = u32;

/// Converts `tobj::Model` vector into vector of internal `Model` representations.
/// It expects valid `tobj::Model` representation, eg. number of positions
/// divisible by 3.
pub fn tobj_to_internal(tobj_models: Vec<tobj::Model>) -> Vec<Model> {
    let mut models = Vec::with_capacity(tobj_models.len());

    for model in tobj_models {
        let mut vertices = Vec::with_capacity(model.mesh.positions.len() / 3);

        for (index, _) in model.mesh.positions.iter().enumerate().step_by(3) {
            vertices.push(Vertex {
                position: [
                    model.mesh.positions[index],
                    model.mesh.positions[index + 1],
                    model.mesh.positions[index + 2],
                ],
            });
        }

        models.push(Model {
            name: model.name,
            vertices,
            indices: model.mesh.indices,
        });
    }

    models
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_tobj_model(indices: Vec<u32>, positions: Vec<f32>) -> tobj::Model {
        tobj::Model {
            name: String::from("Test model"),
            mesh: tobj::Mesh {
                indices,
                positions,
                material_id: None,
                normals: vec![],
                texcoords: vec![],
            },
        }
    }

    #[test]
    fn test_tobj_to_internal_returns_correct_representation_for_single_model() {
        let tobj_model = create_tobj_model(vec![1, 2], vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let tobj_models = vec![tobj_model.clone()];
        let models = tobj_to_internal(tobj_models);

        assert_eq!(
            models,
            vec![Model {
                name: tobj_model.name,
                vertices: vec![
                    Vertex {
                        position: [6.0, 5.0, 4.0]
                    },
                    Vertex {
                        position: [3.0, 2.0, 1.0]
                    }
                ],
                indices: tobj_model.mesh.indices,
            }]
        );
    }

    #[test]
    fn test_tobj_to_internal_returns_correct_representation_for_multiple_models() {
        let tobj_model_1 = create_tobj_model(vec![1, 2], vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let tobj_model_2 = create_tobj_model(vec![3, 4], vec![16.0, 15.0, 14.0, 13.0, 12.0, 11.0]);
        let tobj_models = vec![tobj_model_1.clone(), tobj_model_2.clone()];
        let models = tobj_to_internal(tobj_models);

        assert_eq!(
            models,
            vec![
                Model {
                    name: tobj_model_1.name,
                    vertices: vec![
                        Vertex {
                            position: [6.0, 5.0, 4.0]
                        },
                        Vertex {
                            position: [3.0, 2.0, 1.0]
                        }
                    ],
                    indices: tobj_model_1.mesh.indices,
                },
                Model {
                    name: tobj_model_2.name,
                    vertices: vec![
                        Vertex {
                            position: [16.0, 15.0, 14.0]
                        },
                        Vertex {
                            position: [13.0, 12.0, 11.0]
                        }
                    ],
                    indices: tobj_model_2.mesh.indices,
                }
            ]
        );
    }
}
