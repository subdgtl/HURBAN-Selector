use std::slice;
use std::u32;

use crate::convert::{cast_u32, cast_usize};

use crate::geometry::{self, Geometry};

pub struct ShrinkWrapParams {
    pub geometry: Geometry,
    pub sphere_density: u32,
    pub step: u32,
}

struct ShrinkWrapState {
    sphere_geometry: Geometry,
    current_step: u32,
}

pub struct ShrinkWrapOp {
    params: ShrinkWrapParams,
    state: Option<ShrinkWrapState>,
}

impl ShrinkWrapOp {
    pub fn new(params: ShrinkWrapParams) -> Self {
        Self {
            params,
            state: None,
        }
    }

    pub fn next_value(&mut self) -> Option<Geometry> {
        // Note that the operation does a nontrivial amount of work
        // for its initialization. To make things more flexible for
        // the interpreter (it might not want to start polling the
        // operation right away), we currently have a "cheap
        // constructor" rule - If an operation has computation heavy
        // init, we delay it until needed.
        // FIXME: this constraint may go away, in which case delete
        // this comment :)

        if self.state.is_none() {
            self.init();
        }

        let state = self
            .state
            .as_mut()
            .expect("State must be initialized in `init()`");

        let sphere_geometry_vertices_len = cast_u32(state.sphere_geometry.vertices().len());
        if state.current_step < sphere_geometry_vertices_len {
            let next_current_step = if self.params.step == 0 {
                sphere_geometry_vertices_len
            } else {
                u32::min(
                    state.current_step + self.params.step,
                    sphere_geometry_vertices_len,
                )
            };

            let range = cast_usize(state.current_step)..cast_usize(next_current_step);
            for vertex in &mut state.sphere_geometry.vertices_mut()[range] {
                if let Some(closest) = geometry::find_closest_point(vertex, &self.params.geometry) {
                    vertex.coords = closest.coords;
                }
            }

            state.current_step = next_current_step;

            Some(state.sphere_geometry.clone())
        } else {
            None
        }
    }

    fn init(&mut self) {
        let (center, radius) =
            geometry::compute_bounding_sphere(slice::from_ref(&self.params.geometry));
        let sphere_geometry = geometry::uv_sphere(
            center.coords.into(),
            radius,
            self.params.sphere_density,
            self.params.sphere_density,
        );

        self.state = Some(ShrinkWrapState {
            sphere_geometry,
            current_step: 0,
        });
    }
}
